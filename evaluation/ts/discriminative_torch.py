import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import copy
import os

import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data import get_dataset_config, get_dataset
from evaluation.eval_utils import train_latent_ddim, save_latent
from models import LatentDiff, DiffSDAPriorKarrasTS
from run import parse_args
from sampling import LatentDiffusionProcess, DiffusionProcess
from paths import FINAL_WEIGHTS_ROOT
from utils import generate_exp_string, seed_everything


def discriminative_score_metrics(ori_data, generated_data, args):
    # Basic Parameters
    ori_data, generated_data = torch.Tensor(ori_data), torch.Tensor(generated_data)
    ## Builde a post-hoc RNN discriminator network
    # Network parameters
    hidden_dim = int(args.input_size / 2)
    iterations = 2000
    batch_size = 8

    device = args.device

    class Discriminator(nn.Module):
        def __init__(self, inp_dim, hidden_dim):
            super(Discriminator, self).__init__()

            # the input dim: [batch,channel,length]
            # self.enc_net = nn.Sequential(
            #     BatchLinearUnit(fft_size // 2 + 1, fft_size // 2 + 1, nonlinearity=nn.Tanh()))

            # tensor should be [b,l,c]
            self.rnn = nn.GRU(input_size=inp_dim, hidden_size=hidden_dim, bidirectional=False,
                              num_layers=1, batch_first=True)

            self.linear = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, last_hidden_state = self.rnn(x)
            y_hat_logit = self.linear(last_hidden_state)
            y_hat = nn.functional.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat

    model = Discriminator(args.input_channels, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)

    train_loss = 0.0

    model.train()
    # Training step
    for itt in range(iterations):
        # Batch setting
        X_mb = torch.stack(batch_generator(train_x, batch_size)).to(device)
        X_hat_mb = torch.stack(batch_generator(train_x_hat, batch_size)).to(device)

        y_logit_real, y_pred_real = model(X_mb.float())
        y_logit_fake, y_pred_fake = model(X_hat_mb.float())

        real_labels = torch.ones_like(y_logit_real)
        fake_labels = torch.zeros_like(y_logit_fake)

        d_loss_real = nn.functional.binary_cross_entropy_with_logits(y_logit_real, real_labels).mean()
        d_loss_fake = nn.functional.binary_cross_entropy_with_logits(y_logit_fake, fake_labels).mean()

        d_loss = d_loss_real + d_loss_fake

        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()

        train_loss += d_loss.cpu().item()

    #     if itt % 100 == 0:
    #         # record the loss
    #         print('{}: train loss: {:.4f}'.format(itt, d_loss.cpu().item()))
    #
    #
    # print('final train loss: {:.4f}'.format(train_loss / iterations))

    model.eval()
    with (torch.no_grad()):
        test_x = torch.stack(test_x).to(device)
        test_x_hat = torch.stack(test_x_hat).to(device)
        _, y_pred_real_curr = model(test_x.float())
        _, y_pred_fake_curr = model(test_x_hat.float())

        y_pred_real_curr = y_pred_real_curr.detach().cpu().numpy()
        y_pred_fake_curr = y_pred_fake_curr.detach().cpu().numpy()

        y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
        y_label_final = np.concatenate(
            (np.ones([y_pred_real_curr.shape[1], ]), np.zeros([y_pred_fake_curr.shape[1], ])),
            axis=0)

        # Compute the accuracy
        acc = accuracy_score(y_label_final, (y_pred_final > 0.5).reshape(-1))
        discriminative_score = np.abs(0.5 - acc)

    return discriminative_score


def train_test_divide(data_x, data_x_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat


def batch_generator(data, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)

    return X_mb

def get_latent_model(args, device, logger, model, current_epoch):
    shape = get_dataset_config(args)
    root = f'{args.model_folder}'
    root = os.path.join(root, 'timediffpriorkarras')
    root = os.path.join(root, generate_exp_string(args))
    os.makedirs(root + '_latent', exist_ok=True)
    latent_model_path = os.path.join(root + '_latent', f'model-{args.epochs_latent}.pth')
    args.is_latent = True
    args.deterministic = True
    num_layers = 10
    if os.path.exists(latent_model_path):
        new_shape = (1, args.s_dim + args.d_dim * shape[0], args.s_dim + args.d_dim * shape[0])
        model2 = LatentDiff(args, device, new_shape, num_layers=num_layers)
        model2.load_state_dict(torch.load(latent_model_path))
    else:
        if current_epoch < args.save_epochs:
            return
        copy_args = copy.deepcopy(args)
        number_of_saves = current_epoch // copy_args.save_epochs
        epochs = number_of_saves * copy_args.save_epochs
        args.epoch = epochs
        args.mode = 'save_latent'
        path_to_latent = "{}_{}_latent.npz".format(args.model, generate_exp_string(args).replace(".", "_"))
        batch_size_old = args.batch_size
        if not os.path.exists(os.path.join(os.getcwd(), path_to_latent)):
            args.batch_size = 2
            save_latent(args, device, model)
            args.batch_size = batch_size_old
        args.save_epochs = args.epochs_latent
        args.mode = 'train_latent_ddim'
        args.batch_size = 8
        model2 = train_latent_ddim(args, logger, num_layers=num_layers)
        args.batch_size = batch_size_old
    model2.eval()
    model2.requires_grad_(False)
    model2 = model2.to(device)
    return model, model2, args, device


args_strings = {
    'physionet': '''--mode train --r_seed 42 --dataset physionet --model timediffpriorkarras --s_dim 24 --d_dim 2 --hidden_dim 96 --diffusion_steps 24 --batch_size 128 --learning_rate 5e-5 --mlp_hidden_dim 256 --mlp_hidden_dim_enc 96 --ch_mult 1 2 2 2''',
    'airq':      '''--mode train --r_seed 42 --dataset airq --model timediffpriorkarras --s_dim 16 --d_dim 4 --hidden_dim 512 --diffusion_steps 16 --batch_size 128 --learning_rate 1e-4 --mlp_hidden_dim 256 --mlp_hidden_dim_enc 128 --ch_mult 1 2 2 2''',
    'etth':      '''--mode train --r_seed 42 --dataset etth --model timediffpriorkarras --s_dim 16 --d_dim 4 --hidden_dim 512 --diffusion_steps 32 --batch_size 128 --learning_rate 1e-4 --mlp_hidden_dim 128 --mlp_hidden_dim_enc 256 --ch_mult 1 2 2 2'''
}

weights_path = {
    'physionet': os.environ.get(
        'DIFFSDA_TS_PHYSIONET_WEIGHTS',
        os.path.join(FINAL_WEIGHTS_ROOT, 'timeseries', 'physionet.pth'),
    ),
    'airq': os.environ.get(
        'DIFFSDA_TS_AIRQ_WEIGHTS',
        os.path.join(FINAL_WEIGHTS_ROOT, 'timeseries', 'airq.pth'),
    ),
    'etth': os.environ.get(
        'DIFFSDA_TS_ETTH_WEIGHTS',
        os.path.join(FINAL_WEIGHTS_ROOT, 'timeseries', 'etth.pth'),
    ),
}


def load_model(args):
    shape = get_dataset_config(args)
    model = DiffSDAPriorKarrasTS(args, args.device, shape, ch_mult=args.ch_mult)
    state_dict = torch.load(weights_path[args.dataset])
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    return model

if __name__ == '__main__':
    args_string = args_strings['physionet']
    args = parse_args(args_string.split())
    seed_everything(args.r_seed)
    args.device = device = args.rank
    model = load_model(args)
    model = model.to(device)
    dataloader, validloader, evalloader = get_dataset(args)
    model, model2, args, device = get_latent_model(args, device, None, model, 200)

    metric_iteration = 10
    ## for deterministic results

    gen_sig = []
    real_sig = []
    shape = get_dataset_config(args)
    process_latent = LatentDiffusionProcess(args, model2, device)
    process = DiffusionProcess(args, model, device, shape)
    with torch.no_grad():
        for data in tqdm(evalloader):
            # sample from the model
            b, v = data[0].shape[0], data[0].shape[1]
            og_s, og_d, og_a = model.encoder(data[0].to(device))
            # sample_a = process_latent.sampling(sampling_number=data[0].shape[0])
            # sample_s, sample_d = sample_a[:, :args.s_dim], sample_a[:, args.s_dim:]
            # sample_s = sample_s.unsqueeze(1).expand(b, v, args.s_dim)
            # sample_d = sample_d.view(b, v, args.d_dim)
            # a = torch.cat([sample_d, sample_s], dim=-1).to(device)
            x_T = process.reverse_sampling(data[0].to(device), og_a)
            x_ts = process.sampling(sampling_number=data[0].shape[0], xT=x_T, a=og_a)
            x_ts_no_xt = process.sampling(sampling_number=data[0].shape[0], xT=None, a=og_a)
            print('mse', torch.nn.functional.mse_loss(x_ts, data[0].to(device)))
            print('mse no xt', torch.nn.functional.mse_loss(x_ts_no_xt, data[0].to(device)))
            # --- convert to time series --

            # special case for temperature_rain dataset

            # gen_sig.append(x_ts.detach().cpu().numpy())
            # real_sig.append(data[0].detach().cpu().numpy())

    # gen_sig = np.vstack(gen_sig)
    # real_sig = np.vstack(real_sig)
    # disc_res = []
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # for _ in range(metric_iteration):
    #     dsc = discriminative_score_metrics(real_sig, gen_sig, args)
    #     disc_res.append(dsc)
    # print(f"Discriminative score: {np.mean(disc_res)} and std: {np.std(disc_res)}")
