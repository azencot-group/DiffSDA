import copy
import os

import torch
from torch import nn
import numpy as np

class LinearClassifier(nn.Module):
    def __init__(self, fc_input_dim, n_classes, regression=False):
        super(LinearClassifier, self).__init__()
        self.fc_input_dim = fc_input_dim
        if regression:
            self.n_classes = 1
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_dim, self.n_classes),
                nn.ReLU(),
                nn.Linear(self.n_classes, 1),
                nn.ReLU()
            )
        else:
            self.n_classes = n_classes
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_dim, self.n_classes),
                nn.ReLU(),
                nn.Linear(self.n_classes, self.n_classes))


    def forward(self, f_post, z_post, args):
        if args.category=='global':
            probs = self.classifier(f_post)
        else:
            probs = self.classifier(z_post)

        return probs


def train_classifier(trainset, validset, classifier_model, rep_model, n_epochs, lr, data, args, device):
    """Train a classifier to classify the subgroup of time series"""
    losses_train, losses_val, acc_train, acc_val = [], [], [], []
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    validation_loss = float('inf')
    best_model = copy.deepcopy(classifier_model.state_dict())
    for epoch in range(n_epochs+1):
        train_loss, train_acc = run_classifier_epoch(args, classifier_model, rep_model, trainset, optimizer= optimizer, data=data, train=True, device=device)
        valid_loss, valid_acc = run_classifier_epoch(args, classifier_model, rep_model, validset, optimizer=optimizer, data=data, train=False, device=device)
        losses_train.append(train_loss)
        acc_train.append(train_acc)
        losses_val.append(valid_loss)
        acc_val.append(valid_acc)

        if valid_loss < validation_loss:
            validation_loss = valid_loss
            best_model = copy.deepcopy(classifier_model.state_dict())


        if epoch % 5 == 0:
            print('=' * 30)
            print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
            print("Training loss = %.3f \t Training accuracy = %.3f" % (train_loss, train_acc))
            print("Validation loss = %.3f \t Validation accuracy = %.3f" % (valid_loss, valid_acc))
    return best_model


def run_classifier_epoch(args, classifier_model, rep_model, dataset, data, optimizer=None, train=False, repeat=5, device='cuda'):
    """Training epoch of a classifier"""
    ce_loss = nn.CrossEntropyLoss()
    epoch_loss, epoch_acc= [], []
    for _ in range(repeat):
        for i, data_a in enumerate(dataset):
            x_seq, mask_seq, x_lens = data_a[0].to(device), data_a[1].to(device), data_a[2]
            mask_seq = mask_seq.to(torch.float32)
            if args.category=='global':
                rnd_t = np.random.randint(0, ((x_seq.shape[1] // args.window_size if x_lens is None else min(x_lens)) // args.window_size) - 1)

                if data=='airq':
                    labels = torch.tensor([int(m.item()) - 1 for m in data_a[4][:, 1]], dtype=torch.int64)

                elif data=='physionet':
                    labels = data_a[4][:, 3] - 1
                    labels = labels.to(torch.int64)

            elif args.category=='local':
                rnd_t = np.random.randint(0, ((x_seq.shape[1] if x_lens is None else min(x_lens))//args.window_size)-1)

            with torch.no_grad():
                f_post, z_post, _ = rep_model.encoder(x_seq)#, mask_seq)
            f_post = f_post.detach()
            z_post = z_post.detach()

            if train:
                classifier_model.train()
                optimizer.zero_grad()
                predictions = classifier_model(f_post, z_post, args)
                labels = labels.to(predictions.device)
                loss = ce_loss(predictions, labels)
                loss.backward()
                optimizer.step()
            else:
                classifier_model.eval()
                with torch.no_grad():
                    predictions = classifier_model(f_post, z_post, args)
                labels = labels.to(predictions.device)
                loss = ce_loss(predictions, labels)

            accuracy = (labels == predictions.argmax(dim=-1)).float()
            accuracy = accuracy.cpu().numpy()
            accuracy = np.mean(accuracy)

            epoch_loss.append(loss.detach().cpu().item())
            epoch_acc.append(accuracy)
    return np.mean(epoch_loss), np.mean(epoch_acc)


def classification_exp(representation_classifier, rep_model, args, data, datasets, device='cuda'):
    """Run the classification experiment"""
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]

    best_model = train_classifier(trainset, validset, classifier_model=representation_classifier, rep_model=rep_model, n_epochs= 40 if args.dataset == 'physionet' else 60,
                                  lr=args.learning_rate, data=data, args=args, device=device)

    # Load classifier model weights
    representation_classifier.load_state_dict(best_model)

    test_loss, test_acc = run_classifier_epoch(args, representation_classifier, rep_model, testset, data=data, train=False)
    return test_acc, test_loss

def train_and_test_classifier(args, trainset, validset, testset, rep_model, device):

    rep_model.eval()  # Set the model to evaluation mode

    test_accuracies, test_losses = [], []
    if args.category == 'global':
        classifier_model = LinearClassifier(args.s_dim, n_classes=args.n_classes)
    else:
        classifier_model = LinearClassifier(args.d_dim, n_classes=args.n_classes)

    classifier_model.to(device)

    n_cv = int(os.environ.get('DIFFSDA_TS_CLS_CV', 2))
    for cv in range(n_cv):
        test_acc, test_loss = classification_exp(classifier_model, rep_model, args, data=args.dataset, datasets=(trainset, validset, testset), device=device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

    return test_accuracies, test_losses



