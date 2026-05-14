"""Entry point for training on audio / speech datasets.

Supported datasets: timit, libri

Example
-------
TIMIT (spectrogram)::

    python train_audio.py --dataset timit --model timediffpriorkarras \\
        --mode train --epochs 200 --batch_size 64 \\
        --d_dim 32 --s_dim 32 --hidden_dim 256 \\
        --diffusion_steps 1000 --learning_rate 1e-4 \\
        --fft_size 400 --w_len 320 --h_len 165

LibriSpeech (mel spectrogram)::

    python train_audio.py --dataset libri --model timediffpriorkarras \\
        --mode train --epochs 200 --batch_size 64 --mel \\
        --d_dim 32 --s_dim 32 --hidden_dim 256 \\
        --diffusion_steps 1000 --learning_rate 1e-4
"""

import gc

import torch
import torch.multiprocessing as mp
import torchaudio
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from data import get_dataset, get_dataset_config
from datasets_util.LibriSpeech import libri_normalize
from distributed import (
    call_barrier_if_distributed, ddp_setup, gather_total_loss, get_epoch_range,
    get_models_if_dist, is_main_proc, next_free_port,
)
from evaluation.eval_utils import save_timit_audio, save_timit_audio_swap
from HiFiGAN import audio_path_to_data_hifi, mel_spectrogram
from loggers import TqdmLogger
from models import LossWrapper, DiffSDAPriorKarrasTimit
from run import log_loss, parse_args, train_step
from sampling import DiffusionProcess
from timit_utils import audio_path_to_data, timit_normalize, voice_verification_mean
from utils import GradualWarmupScheduler, save_model, seed_everything
from vq_models.vq_configs import get_fs_model

AUDIO_DATASETS = ['timit', 'libri']


def parse_audio_args(text=None):
    args = parse_args(text)
    if args.dataset not in AUDIO_DATASETS:
        raise ValueError(
            f'Dataset "{args.dataset}" is not an audio dataset. '
            f'Choose from: {AUDIO_DATASETS}'
        )
    return args


def _train_sampling(args, current_epoch, device, evalloader, logger, model, shape, vq_model,
                    spectrogram, resampler):
    if args.rank == 0:
        model.eval()
        process = DiffusionProcess(args, model, device, shape)
        data = next(iter(evalloader))

        if args.mel:
            hifigan, _, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
            hifigan = hifigan.to(device)
            denoiser = denoiser.to(device)
        else:
            hifigan, denoiser = None, None

        if args.mel:
            data, ch_num = audio_path_to_data_hifi(data['wav'], resampler)
        else:
            data, ch_num = audio_path_to_data(data['wav'])
        data = spectrogram(data.to(device))
        data = data.permute(0, 2, 1)
        if args.dataset == 'timit':
            data = timit_normalize(args, data)
        elif args.dataset == 'libri':
            data = libri_normalize(args, data)

        save_timit_audio(args, model, data, ch_num, device, shape, logger=logger, process=process,
                         hifigan=hifigan, denoiser=denoiser)
        save_timit_audio(args, model, data, ch_num, device, shape, logger=logger, process=process,
                         use_xt=True, hifigan=hifigan, denoiser=denoiser)
        save_timit_audio_swap(args, model, data, ch_num, device, shape, logger=logger, process=process,
                              hifigan=hifigan, denoiser=denoiser)
        save_timit_audio_swap(args, model, data, ch_num, device, shape, logger=logger, process=process,
                              use_xt=True, hifigan=hifigan, denoiser=denoiser)

        eer_static, eer_dynamic = voice_verification_mean(args, model, spectrogram, evalloader, resampler)
        logger.log('eval/eer_static', eer_static)
        logger.log('eval/eer_dynamic', eer_dynamic)
        logger.log('eval/eer_gap', eer_dynamic - eer_static)

        model.train()


def train_sampling(args, current_epoch, dataloader, device, evalloader, logger, model, shape, train_model,
                   vq_model, spectrogram, resampler):
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        if current_epoch % args.sample_epochs == 0:
            model, _ = get_models_if_dist(args, model, train_model, vq_model)
            if args.gpu_num > 1:
                dataloader.sampler.set_epoch(current_epoch)
            _train_sampling(args, current_epoch, device, evalloader, logger, model, shape, vq_model,
                            spectrogram, resampler)
            call_barrier_if_distributed(args)
            gc.collect()
            torch.cuda.empty_cache()


def train(args):
    logger = TqdmLogger(rank=args.rank)
    if args.tags is not None:
        logger.add_tags(args.tags)
    seed_everything(args.r_seed)
    args.device = device = args.rank

    if args.first_stage_model:
        vq_model = get_fs_model(args, device)
        args.input_channels = args.z_channels_vq
    else:
        vq_model = None
        args.vq_input_size = args.input_size

    shape = get_dataset_config(args)
    logger.log_hparams(dict(vars(args)))

    if args.mel:
        resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
        spectrogram = mel_spectrogram
    else:
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=args.fft_size, win_length=args.w_len, hop_length=args.h_len,
            power=args.power).to(device)
        resampler = None

    dataloader, evalloader = get_dataset(args)

    model = DiffSDAPriorKarrasTimit(args, device, shape, ch_mult=args.ch_mult)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    train_model = LossWrapper(model, None)
    if args.gpu_num > 1:
        train_model = DDP(train_model, device_ids=[args.rank], find_unused_parameters=True)

    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2., warm_epoch=1, after_scheduler=cosineScheduler)

    global_step = 0
    to_log = {}
    loss_acc = 0
    epoch_range = get_epoch_range(args)
    for curr_epoch in epoch_range:
        logger.log_name_params('train/epoch', curr_epoch)
        total_loss = 0
        batch_bar = tqdm(dataloader, desc="Batch #", unit_scale=args.gpu_num) if is_main_proc(args) else dataloader
        for idx, data in enumerate(batch_bar):
            if args.mel:
                data, ch_num = audio_path_to_data_hifi(data['wav'], resampler)
            else:
                data, _ = audio_path_to_data(data['wav'])
            data = data.to(device=device)
            data = spectrogram(data)
            data = data.permute(0, 2, 1)
            if args.dataset == 'timit':
                data = timit_normalize(args, data)
            elif args.dataset == 'libri':
                data = libri_normalize(args, data)

            loss = train_step(args, curr_epoch, data, model, optimizer, to_log, train_model)
            loss_acc += loss.mean().detach()
            total_loss += loss.detach().item()
            loss_acc, to_log = log_loss(args, batch_bar, device, global_step, logger, loss_acc, to_log)
            if global_step % args.log_interval == 0 and global_step > 0:
                loss_acc = 0
            global_step += args.gpu_num

        total_loss = gather_total_loss(args, device, total_loss)
        logger.log('train/avg_total_loss', total_loss / (len(dataloader) * args.gpu_num), curr_epoch)
        current_epoch = curr_epoch + 1
        warmUpScheduler.step()
        call_barrier_if_distributed(args)
        if current_epoch % args.save_epochs == 0:
            model, _ = get_models_if_dist(args, model, train_model, vq_model)
            if args.rank == 0:
                save_model(args, current_epoch, model)
        call_barrier_if_distributed(args)
        train_sampling(args, current_epoch, dataloader, device, evalloader, logger, model, shape,
                       train_model, vq_model, spectrogram, resampler)

    logger.stop()


def train_rank(rank, args, port):
    args.rank = rank
    ddp_setup(args.rank, args.gpu_num, port)
    train(args)
    destroy_process_group()


if __name__ == '__main__':
    args = parse_audio_args()
    if args.gpu_num > 1 or args.gpu_num < 1:
        world_size = args.gpu_num if args.gpu_num != -1 else torch.cuda.device_count()
        args.gpu_num = world_size
        port = str(next_free_port())
        args.log_interval = args.gpu_num * args.log_interval
        mp.spawn(train_rank, args=(args, port), nprocs=world_size, join=True)
    else:
        train(args)
