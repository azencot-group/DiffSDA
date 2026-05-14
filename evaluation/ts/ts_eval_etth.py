import copy

import numpy as np
import torch
from torch import nn
import tqdm

class Predictor(nn.Module):
    """Simple classifier layer to classify the subgroup of data

    Args:
        fc_sizes: Hidden size of the predictor MLP. default: [32, 8]
    """

    def __init__(self, fc_sizes=[32, 8], args=None, window_size=None):
        super(Predictor, self).__init__()
        self.window_size = window_size

        # Define the fully connected (fc) layers
        layers = []
        in_features = args.s_dim + args.d_dim
        for out_features in fc_sizes:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.LeakyReLU())
            in_features = out_features
        self.fc = nn.Sequential(*layers)

        # Final probability layer
        self.prob = nn.Linear(in_features=fc_sizes[-1], out_features=1)

        self.relu = nn.LeakyReLU()

        # Batch Normalization Layer
        self.batch_norm = nn.BatchNorm1d(args.s_dim + args.d_dim)  # Apply normalization across the 16 sequences

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, local_encs, global_encs, x_lens):
        # local_encs: [10,12], global_encs: [10, 28, 4], x_lens: [10]
        if global_encs is not None:
            global_encs_expanded = global_encs.unsqueeze(1).repeat(1, local_encs.shape[1], 1)  # [10, 28, 12]
            h = torch.cat([local_encs, global_encs_expanded], dim=2)  # [10, 28, 16]

            # h will have shape: [5, 10, 16]
            h = self.batch_norm((h.transpose(1, 2))).transpose(1, 2)
        else:
            h = local_encs

        logits = self.fc(h)  # [10, 28, 8]
        probs = self.relu(self.prob(logits)).squeeze(-1)  # [10, 28, 1]
        if self.window_size:
            probs = block_pred(probs, self.window_size)
        return probs  # [5, 10]


def block_labels(data_loader, args):
    window_size = args.window_size

    # List to store the selected labels from each batch
    selected_labels_list = []

    # Loop through each batch in the data loader
    for batch in data_loader:
        # Get the selected labels from the current batch
        selected_labels = batch[3][:, :, 1]
        selected_labels_list.append(selected_labels)

    # Concatenate all the selected labels along the first dimension
    all_labels_blocks = torch.cat(selected_labels_list, dim=0)

    # Split the tensor along the 1-axis (time dimension)
    all_labels_blocks = torch.split(all_labels_blocks, window_size, dim=1)

    # Calculate the average of each block along the 1-axis (time dimension)
    # and stack them along the last dimension
    all_labels_blocks = torch.stack([torch.mean(block, dim=1) for block in all_labels_blocks], dim=-1)

    return all_labels_blocks

def block_pred(pred, window_size):

    pred_blocks = torch.split(pred, window_size, dim=1)
    pred_blocks = torch.stack([torch.mean(block, dim=1) for block in pred_blocks], dim=-1)
    return pred_blocks


def run_epoch_predictor(args, predictor_model, data_loader, rep_model, optimizer=None, label_blocks=None, train=True, test=False, trainable_vars=None, device='cuda'):
    """Training epoch for training the classifier"""
    mae = nn.L1Loss()
    epoch_loss = []

    b_start = 0
    for i, data in enumerate(data_loader):
        x_seq, mask_seq, x_lens = data[0].to(device), data[1].to(device), data[2].to(device)

        mask_seq = None

        labels = label_blocks[b_start:b_start + len(x_seq)]
        b_start += len(x_seq)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels)

        with torch.no_grad():
            f_post, z_post, _ = rep_model.encoder(x_seq)
        f_post = f_post.detach()
        z_post = z_post.detach()

        lens = (x_lens // args.window_size).int()  # lens.shape = [10]

        if train:
            predictor_model.train()
            optimizer.zero_grad()
            predictions = predictor_model(z_post, f_post, lens)
            labels = labels.float().to(predictions.device)

            loss = mae(labels, predictions)
            loss.backward()
            optimizer.step()
        else:
            predictor_model.eval()
            with torch.no_grad():
                predictions = predictor_model(z_post, f_post, lens)
            labels = labels.to(predictions.device)  # labels.shape = [10, 28]
        epoch_loss.append(mae(labels, predictions).detach().cpu().numpy().mean())
    return np.mean(epoch_loss)


def train_predictor_and_test_avg_oil_temp(args, trainset, validset, testset, rep_model, device):

    rep_model.eval()  # Set the model to evaluation mode

    ##### ----- finished training our model, now train downstream task ----- #####
    test_loss = []
    label_blocks_train = block_labels(trainset, args)
    label_blocks_train.to(device)
    label_blocks_valid = block_labels(validset, args)
    label_blocks_valid.to(device)
    label_blocks_test = block_labels(testset, args)
    label_blocks_test.to(device)

    for cv in range(3):
        predictor_model = Predictor([32, 8], args, args.window_size)
        predictor_model.to(device)
        n_epochs = 300
        lr = 0.001

        optimizer = torch.optim.Adam(predictor_model.parameters(), lr=lr)
        losses_train = []
        losses_val = []
        best_val = float('inf')
        for epoch in range(n_epochs + 1):
            epoch_loss_train = run_epoch_predictor(args, predictor_model, trainset, rep_model, optimizer=optimizer,
                                                   label_blocks=label_blocks_train,
                                                   train=True, test=False, trainable_vars=None, device=device)
            if epoch and epoch % 5 == 0:
                print('=' * 30)
                print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
                losses_train.append(epoch_loss_train)

                epoch_loss_val = run_epoch_predictor(args, predictor_model, validset, rep_model,
                                                     label_blocks=label_blocks_valid, train=False, test=False, device=device)
                # if epoch_loss_val < best_val:
                #     best_val = epoch_loss_val
                losses_val.append(epoch_loss_val)
                te_loss = run_epoch_predictor(args, predictor_model, testset, rep_model, label_blocks=label_blocks_test,
                                              train=False, test=True, device=device)

                print("Training loss = %.3f" % (epoch_loss_train))
                print("Validation loss = %.3f" % (epoch_loss_val))
                print('Test loss =  %.3f' % (te_loss))
        test_loss.append(
            run_epoch_predictor(args, predictor_model, testset, rep_model, label_blocks=label_blocks_test, train=False, device=device))

    return test_loss