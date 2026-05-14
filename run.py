"""Shared training utilities.

Modality-specific entry points:
    train_video.py      — MUG, TAICHI, VoxCeleb1, CelebV-HQ
    train_audio.py      — TIMIT, LibriSpeech
    train_timeseries.py — Air Quality, ETT-h, PhysioNet
"""

import argparse
import os

import torch
from tqdm.auto import tqdm

from paths import MODELS_ROOT, DATASETS_ROOT
from distributed import gather_logs, is_main_proc

import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from distributed import next_free_port, ddp_setup

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------

def parse_args(text=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--r_seed', type=int, default=0,
                        help='the value of given random seed')
    parser.add_argument('--img_id', type=int, default=0,
                        help='the id of given img')
    parser.add_argument('--model', required=True,
                        choices=['diff', 'timediffpriorkarras'],
                        help='which type of model to run')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'save_latent', 'train_latent_ddim', 'eval'], help='which mode to run')
    parser.add_argument('--dataset', required=True,
                        choices=['libri', 'airq', 'etth', 'physionet', 'timit', 'mug', 'taichi', 'vox1', 'celebv'],
                        help='training dataset')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('-el', '--epochs_latent', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--save_epochs', type=int, default=5,
                        help='number of epochs to save model')
    parser.add_argument('--sample_epochs', type=int, default=50,
                        help='number of epochs to sample images from the model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--optimizer', default='adam', choices=['adam'],
                        help='optimization algorithm')
    parser.add_argument('--model_folder', default=MODELS_ROOT,
                        help='folder where model checkpoints are saved (env: DIFFSDA_MODELS_ROOT)')
    parser.add_argument('--deterministic', action='store_true',
                        default=False, help='deterministid sampling')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='number of input channels')
    parser.add_argument('--unets_channels', type=int, default=64,
                        help='number of input channels')
    parser.add_argument('--encoder_channels', type=int, default=64,
                        help='number of input channels')
    parser.add_argument('--input_size', type=int, default=32,
                        help='expected size of input')
    parser.add_argument('--a_dim', type=int, default=32,
                        help='dimensionality of auxiliary variable')
    parser.add_argument('--d_dim', type=int, default=32,
                        help='dimensionality of dynamic variable')
    parser.add_argument('--s_dim', type=int, default=32,
                        help='dimensionality of static variable')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='dimensionality of bottleneck lstm')
    parser.add_argument('--mlp_hidden_dim', type=int, default=201,
                        help='dimensionality of mlp hidden layers')
    parser.add_argument('--mlp_hidden_dim_enc', type=int, default=201,
                        help='dimensionality of mlp hidden layers')
    parser.add_argument('--beta1', type=float, default=1e-5,
                        help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2,
                        help='value of beta T')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='number of diffusion steps')
    parser.add_argument('--diffusion_steps_latent', type=int, default=1000,
                        help='number of diffusion steps for latent diffusion')
    parser.add_argument('--sampling_number', type=int, default=4,
                        help='number of sampled images')
    parser.add_argument('--data_dir', type=str, default=DATASETS_ROOT,
                        help='root dir that contains all datasets (env: DIFFSDA_DATASETS_ROOT)')
    parser.add_argument('--is_latent', action='store_true',
                        help='use latent diffusion for unconditional sampling.')
    parser.add_argument('--video_length', type=int, default=15,
                        help='video length for video dataset.')
    parser.add_argument('--eval_fix', action='store_true',
                        help='will test the model at the end of training on fixed latent variable, using ldm model')
    parser.add_argument('--shared_noise', action='store_true', help='make the same noise for all video series')
    parser.add_argument('--fp16', action='store_true', help='enable fp16 training')
    parser.add_argument('--tags', type=str, default=None, help='free-form run tags (recorded in log output)', nargs='+')
    parser.add_argument('--ch_mult', type=int, default=[1, 2, 2, 2], help='ch mut', nargs='+')
    parser.add_argument('--attn', type=int, default=[2], help='attn', nargs='+')
    parser.add_argument('--swap_num_samples', type=int, default=-1,
                        help='number of sample to generate. NOTE: the real number will be 32//bz, in last epoh it will be the all dataset')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval')
    parser.add_argument('--gpu_num', type=int, default=1, help='gpu number')
    parser.add_argument('--rank', type=int, default=0, help='rank number of process')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--scale', action='store_true', help='normalize the latent input')
    parser.add_argument('--use_pre_split', action='store_true', help='use pre split')
    parser.add_argument('--newsplit', action='store_true', help='use new split')
    parser.add_argument('--extract_speed', type=int, default=-1, help='extract speed')
    parser.add_argument('--sheared_s', action='store_false', help='sheared s')

    ##### VQ PARAMS
    parser.add_argument('--first_stage_model', type=str, default=None, choices=['vq4', 'vq8', 'kl8', 'vq8ft', None],
                        help='first stage model')
    parser.add_argument('--resolution_vq', type=int, default=256, help='resolution')
    parser.add_argument('--z_channels_vq', type=int, default=4, help='z_channels')
    parser.add_argument('--img_channels', type=int, default=3, help='z_channels')
    parser.add_argument('--ch_vq', type=int, default=128, help='ch')
    parser.add_argument('--out_ch_vq', type=int, default=3, help='out ch')
    parser.add_argument('--ch_mult_vq', type=int, default=[1, 2, 4], help='ch mut', nargs='+')
    parser.add_argument('--vq_input_size', type=int, default=32, help='quantized input size')
    parser.add_argument('--latent_dataset', action='store_true', help='use latent dataset')
    parser.add_argument('--latent_s_d_split', action='store_true', help='two latent model for s and d')
    parser.add_argument('--latent_const', action='store_true', help='use l1 loss and const scheduler')

    # audio params
    parser.add_argument('--w_len', type=int, default=320, help='window length')
    parser.add_argument('--h_len', type=int, default=165, help='hop length')
    parser.add_argument('--power', type=int, default=1, help='power of the spectrogram')
    parser.add_argument('--fft_size', type=int, default=400, help='power of the spectrogram')
    parser.add_argument('--no_time', action='store_true', help='make timit run a unet on full frame')
    parser.add_argument('--mel', action='store_true', help='use mel spectrogram')

    # TS params
    parser.add_argument('--category', default='global', type=str, help='global or local')
    parser.add_argument('--n_classes', type=int, default=1, help='for classification test for TS')
    parser.add_argument('--window_size', type=int, default=10, help='For TS models')
    parser.add_argument('--ckpt', type=str, default='./trained_models/')

    if text is not None:
        args = parser.parse_args(text)
    else:
        args = parser.parse_args()

    return args


def train_step(args, curr_epoch, data, model, optimizer, to_log, train_model):
    optimizer.zero_grad()
    loss = train_model(args=args, x=data, curr_epoch=curr_epoch)
    if len(loss) == 2:
        loss, to_sum = loss
        for key, value in to_sum.items():
            if key not in to_log:
                to_log[key] = 0
            to_log[key] += value
    if torch.isnan(loss).any():
        raise ValueError('nan loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss


def log_loss(args, batch_bar, device, global_step, logger, loss_acc, to_log):
    if global_step % args.log_interval == 0 and global_step > 0:
        loss_acc, to_log = gather_logs(args, device, loss_acc, to_log)
        if is_main_proc(args):
            batch_bar.set_postfix(loss=format(loss_acc / args.log_interval, '.4f'))
            logger.log('train/loss', loss_acc / args.log_interval, global_step)
            for key, value in to_log.items():
                logger.log(f'train/{key}', value / args.log_interval, global_step)
            loss_acc = 0
        for key in to_log.keys():
            to_log[key] = 0
    return loss_acc, to_log
