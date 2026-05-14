"""Entry point for training on video datasets.

Supported datasets: mug, taichi, vox1, celebv

Example
-------
Single GPU::

    python train_video.py --dataset vox1 --model timediffpriorkarras \\
        --mode train --epochs 200 --batch_size 16 --input_size 256 \\
        --first_stage_model vq8ft --latent_dataset \\
        --d_dim 128 --s_dim 128 --hidden_dim 512 \\
        --diffusion_steps 32 --learning_rate 1e-4

Multi-GPU (DDP)::

    python train_video.py --dataset vox1 --model timediffpriorkarras \\
        --mode train --gpu_num 4 --batch_size 4 ...
"""

import gc

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from data import get_dataset, get_dataset_config, get_latent
from distributed import (
    call_barrier_if_distributed, ddp_setup, gather_total_loss, get_epoch_range,
    get_models_if_dist, is_main_proc, next_free_port,
)
from evaluation.eval_utils import log_swap_sample, sample_cond_images
from loggers import TqdmLogger
from models import LossWrapper, DiffSDAPriorKarras
from run import log_loss, parse_args, train_step
from sampling import DiffusionProcess
from utils import GradualWarmupScheduler, get_model_path, save_model, seed_everything
from vq_models.vq_configs import get_fs_model

VIDEO_DATASETS = ['mug', 'taichi', 'vox1', 'celebv']


def parse_video_args(text=None):
    args = parse_args(text)
    if args.dataset not in VIDEO_DATASETS:
        raise ValueError(
            f'Dataset "{args.dataset}" is not a video dataset. '
            f'Choose from: {VIDEO_DATASETS}'
        )
    # Raw video frames are 3-channel; overridden to z_channels_vq in train() when --first_stage_model is set.
    if not args.first_stage_model:
        args.input_channels = 3
    return args


def _train_sampling(args, current_epoch, device, evalloader, logger, model, shape, vq_model):
    if args.rank == 0:
        model.eval()
        process = DiffusionProcess(args, model, device, shape)
        data = next(iter(evalloader))
        data = data[0]
        sample_cond_images(args, model, data, device, shape, logger=logger, epoch=current_epoch,
                           vq_model=vq_model, process=process)
        examples = iter(evalloader)
        log_swap_sample(args, next(examples), model, device, shape, logger=logger, vq_model=vq_model,
                        extra_text='', num_example=args.sampling_number, process=process)
        log_swap_sample(args, next(examples), model, device, shape, logger=logger, vq_model=vq_model,
                        extra_text='use_xt', num_example=args.sampling_number, process=process, use_xt=True)
        model.train()


def train_sampling(args, current_epoch, dataloader, device, evalloader, logger, model, shape, train_model, vq_model):
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        if current_epoch % args.sample_epochs == 0:
            model, _ = get_models_if_dist(args, model, train_model, vq_model)
            if args.gpu_num > 1:
                dataloader.sampler.set_epoch(current_epoch)
            _train_sampling(args, current_epoch, device, evalloader, logger, model, shape, vq_model)
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

    dataloader, evalloader = get_dataset(args)
    if args.first_stage_model and args.latent_dataset:
        dataloader = get_latent(args)

    model = DiffSDAPriorKarras(args, device, shape, ch_mult=args.ch_mult, attn=args.attn)
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
            data = data[0].to(device=device)
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
        train_sampling(args, current_epoch, dataloader, device, evalloader, logger, model, shape, train_model, vq_model)

    logger.stop()


def train_rank(rank, args, port):
    args.rank = rank
    ddp_setup(args.rank, args.gpu_num, port)
    train(args)
    destroy_process_group()


if __name__ == '__main__':
    args = parse_video_args()
    if args.gpu_num > 1 or args.gpu_num < 1:
        world_size = args.gpu_num if args.gpu_num != -1 else torch.cuda.device_count()
        args.gpu_num = world_size
        port = str(next_free_port())
        args.log_interval = args.gpu_num * args.log_interval
        mp.spawn(train_rank, args=(args, port), nprocs=world_size, join=True)
    else:
        train(args)
