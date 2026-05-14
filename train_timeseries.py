"""Entry point for training on time-series datasets.

Supported datasets: airq, etth, physionet

Example
-------
Air Quality::

    python train_timeseries.py --dataset airq --model timediffpriorkarras \\
        --mode train --epochs 200 --batch_size 128 --learning_rate 1e-4 \\
        --s_dim 16 --d_dim 4 --hidden_dim 512 \\
        --diffusion_steps 32 --mlp_hidden_dim 256 --mlp_hidden_dim_enc 128 \\
        --ch_mult 1 2 2 2

ETT-h::

    python train_timeseries.py --dataset etth --model timediffpriorkarras \\
        --mode train --epochs 200 --batch_size 128 --learning_rate 1e-4 \\
        --s_dim 16 --d_dim 4 --hidden_dim 512 \\
        --diffusion_steps 32 --mlp_hidden_dim 128 --mlp_hidden_dim_enc 256 \\
        --ch_mult 1 2 2 2

PhysioNet::

    python train_timeseries.py --dataset physionet --model timediffpriorkarras \\
        --mode train --epochs 200 --batch_size 128 --learning_rate 5e-5 \\
        --s_dim 24 --d_dim 2 --hidden_dim 96 \\
        --diffusion_steps 32 --mlp_hidden_dim 256 --mlp_hidden_dim_enc 96 \\
        --ch_mult 1 2 2 2
"""

import gc

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from data import get_dataset, get_dataset_config
from distributed import (
    call_barrier_if_distributed, ddp_setup, gather_total_loss, get_epoch_range,
    get_models_if_dist, is_main_proc, next_free_port,
)
from loggers import TqdmLogger
from models import LossWrapper, DiffSDAPriorKarrasTS
from run import log_loss, parse_args, train_step
from evaluation.ts.ts_eval_etth import train_predictor_and_test_avg_oil_temp
from evaluation.ts.ts_eval_utils import train_and_test_classifier
from evaluation.ts.ts_eval_utils_predictor import train_predictor_and_test_mortality
from utils import GradualWarmupScheduler, save_model, seed_everything
from vq_models.vq_configs import get_fs_model

TS_DATASETS = ['airq', 'etth', 'physionet']


def parse_ts_args(text=None):
    args = parse_args(text)
    if args.dataset not in TS_DATASETS:
        raise ValueError(
            f'Dataset "{args.dataset}" is not a time-series dataset. '
            f'Choose from: {TS_DATASETS}'
        )
    return args


def _train_sampling(args, current_epoch, device, evalloader, logger, model, shape, vq_model):
    # TS sampling is handled via eval metrics in the train loop; nothing to visualize here.
    pass


def train_sampling(args, current_epoch, dataloader, device, evalloader, logger, model, shape, train_model,
                   vq_model, validloader):
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        if current_epoch % args.sample_epochs == 0:
            model, _ = get_models_if_dist(args, model, train_model, vq_model)
            if args.gpu_num > 1:
                dataloader.sampler.set_epoch(current_epoch)
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

    dataloader, validloader, evalloader = get_dataset(args)

    model = DiffSDAPriorKarrasTS(args, device, shape, ch_mult=args.ch_mult)
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
            data, mask, x_lens, *_ = data
            data, mask = data.to(device=device), mask.to(device=device)
            data = (data, mask, x_lens)

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
                       train_model, vq_model, validloader)

        if args.dataset in ['airq', 'physionet']:
            if current_epoch % args.sample_epochs == 0:
                model, _ = get_models_if_dist(args, model, train_model, vq_model)
                model.eval()
                test_accuracies, test_losses = train_and_test_classifier(
                    args, dataloader, validloader, evalloader, model, device)
                logger.log('eval/accuracy', np.mean(test_accuracies), current_epoch)
                logger.log('eval/loss', np.mean(test_losses), current_epoch)
                if args.dataset == 'physionet':
                    cv_loss, cv_acc, cv_auroc = [], [], []
                    for cv in range(3):
                        test_loss, test_acc, test_auroc = train_predictor_and_test_mortality(
                            args, dataloader, validloader, evalloader, model, device)
                        cv_loss.append(test_loss)
                        cv_acc.append(test_acc)
                        cv_auroc.append(test_auroc)
                    logger.log('eval/cv_loss', np.mean(cv_loss), current_epoch)
                    logger.log('eval/AUPRC', np.mean(cv_acc), current_epoch)
                    logger.log('eval/AUROC', np.mean(cv_auroc), current_epoch)
                model.train()

        if args.dataset == 'etth':
            if current_epoch % args.sample_epochs == 0:
                model, _ = get_models_if_dist(args, model, train_model, vq_model)
                model.eval()
                test_loss = train_predictor_and_test_avg_oil_temp(
                    args, dataloader, validloader, evalloader, model, device)
                logger.log('eval/MAE', np.mean(test_loss), current_epoch)
                model.train()

    logger.stop()


def train_rank(rank, args, port):
    args.rank = rank
    ddp_setup(args.rank, args.gpu_num, port)
    train(args)
    destroy_process_group()


if __name__ == '__main__':
    args = parse_ts_args()
    if args.gpu_num > 1 or args.gpu_num < 1:
        world_size = args.gpu_num if args.gpu_num != -1 else torch.cuda.device_count()
        args.gpu_num = world_size
        port = str(next_free_port())
        args.log_interval = args.gpu_num * args.log_interval
        mp.spawn(train_rank, args=(args, port), nprocs=world_size, join=True)
    else:
        train(args)
