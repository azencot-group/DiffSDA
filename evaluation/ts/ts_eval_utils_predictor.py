import copy
import os

import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class Predictor(nn.Module):
    """Simple classifier layer to classify the subgroup of data

    Args:
        fc_sizes: Hidden size of the predictor MLP. default: [32, 8]
    """

    def __init__(self, rnn_size, fc_sizes, f_dim, z_dim):
        super(Predictor, self).__init__()

        self.rnn_size = rnn_size
        self.fc_sizes = fc_sizes
        self.f_dim = f_dim
        self.z_dim = z_dim

        self.rnn = nn.LSTM(input_size=z_dim, hidden_size=rnn_size, batch_first=True)

        layers = []
        in_features = rnn_size + f_dim  # 44
        for out_features in fc_sizes:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        self.fc = nn.Sequential(*layers)

        self.dropout = nn.Dropout(p=0.5)

        # Final probability layer
        self.prob = nn.Linear(out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_encs, global_encs, x_lens):
        x_lens = x_lens.to(dtype=torch.int32)
        h, _ = self.rnn(local_encs)  # [10, 20, 32]
        h = torch.stack([h[i, x_lens[i] - 1, :] for i in range(len(h))])

        if not global_encs is None:
            h = torch.cat([h, global_encs], dim=-1)

        logits = self.dropout(self.fc(h))
        probs = self.sigmoid(self.prob(logits))
        return probs.squeeze(-1)

def run_epoch_predictor(args, predictor_model, data_loader, rep_model, optimizer=None, train=True, test=False, trainable_vars=None, device='cuda'):
    """Training epoch for training the classifier"""
    bce_loss = nn.BCELoss()
    epoch_loss, epoch_acc, epoch_auroc = [], [], []
    all_labels, all_predictions = [], []

    for i, data in enumerate(data_loader):
        x_seq, mask_seq, x_lens = data[0].to(device), data[1].to(device), data[2].to(device)

        labels = data[4][:, -1]

        with torch.no_grad():
            f_post, z_post, _ = rep_model.encoder(x_seq)
        f_post = f_post.detach()
        z_post = z_post.detach()

        lens = x_lens // args.window_size

        if train:
            predictor_model.train()
            predictor_model.zero_grad()
            predictions = predictor_model(z_post, f_post, lens)
            labels = labels.to(predictions.device)
            loss = bce_loss(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor_model.parameters(), max_norm=1)
            optimizer.step()
        else:
            predictor_model.eval()
            with torch.no_grad():
                predictions = predictor_model(z_post, f_post, lens)
                labels = labels.to(predictions.device)
                loss = bce_loss(predictions, labels)
            labels = labels.to(predictions.device)
        epoch_loss.append(loss.detach().cpu().numpy().mean())
        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    epoch_acc = average_precision_score(all_labels, all_predictions)
    epoch_auroc = (roc_auc_score(all_labels, all_predictions))
    return np.mean(epoch_loss), epoch_acc, epoch_auroc



def train_predictor_and_test_mortality(args, trainset, validset, testset, rep_model, device):

    rep_model.eval()  # Set the model to evaluation mode

    ##### ----- finished training our model, now train downstream task ----- #####
    predictor_model = Predictor(32, [16], args.s_dim, args.d_dim)
    predictor_model.to(device)
    n_epochs = 40
    lr = 1e-3

    optimizer = torch.optim.Adam(predictor_model.parameters(), lr=lr)

    losses_train, acc_train, auroc_train = [], [], []
    losses_val, acc_val, auroc_val = [], [], []
    best_acc_val = 0
    best_model = copy.deepcopy(predictor_model.state_dict())
    for epoch in range(n_epochs + 1):
        epoch_loss_train, epoch_acc_train, epoch_auroc_train = run_epoch_predictor(args, predictor_model, trainset, rep_model, optimizer=optimizer, train=True, test=False, trainable_vars=None, device=device)
        if epoch % 1 == 0:
            print('=' * 30)
            print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
            losses_train.append(epoch_loss_train)
            acc_train.append(epoch_acc_train)
            auroc_train.append(epoch_auroc_train)
            print("Training loss = %.3f \t Accuracy = %.3f \t AUROC = %.3f" % (
                epoch_loss_train, epoch_acc_train, epoch_auroc_train))
            epoch_loss_val, epoch_acc_val, epoch_auroc_val = run_epoch_predictor(args, predictor_model, validset, rep_model, train=False)
            losses_val.append(epoch_loss_val)
            acc_val.append(epoch_acc_val)
            auroc_val.append(epoch_auroc_val)
            print("Validation loss = %.3f \t Accuracy = %.3f \t AUROC = %.3f" % (
                epoch_auroc_val, epoch_acc_val, epoch_auroc_val))
            if epoch_acc_val > best_acc_val:
                best_acc_val = epoch_acc_val
                best_model = copy.deepcopy(predictor_model.state_dict())

    predictor_model.load_state_dict(best_model)
    test_loss, test_acc, test_auroc = run_epoch_predictor(args, predictor_model, testset, rep_model, train=False)
    print("\n Test performance \t loss = %.3f \t AUPRC = %.3f \t AUROC = %.3f" % (
        test_loss, test_acc, test_auroc))

    return test_loss, test_acc, test_auroc


