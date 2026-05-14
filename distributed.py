import os
import random
import socket
from datetime import timedelta

import torch
from torch.distributed import init_process_group, all_gather_object, all_gather, barrier
from tqdm.asyncio import trange


def next_free_port( port=12355, max_port=65535 ):
    rand_offset = random.randint(0, 1000)
    port += rand_offset
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')


def ddp_setup(rank, world_size, port="12355"):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=60))
    torch.cuda.set_device(rank)


def gather_total_loss(args, device, total_loss):
    if args.gpu_num <= 1:
        return total_loss
    total_loss_list = [torch.zeros(1).to(device) for _ in range(args.gpu_num)]
    all_gather_object(total_loss_list, total_loss)
    if args.rank == 0:
        total_loss = sum(total_loss_list) / args.gpu_num
    return total_loss


def gather_logs(args, device, loss_acc, to_log):
    if args.gpu_num <= 1:
        return loss_acc, to_log
    loss_list = [torch.zeros(1).to(device) for _ in range(args.gpu_num)]
    all_gather(loss_list, loss_acc)
    to_log_list = [None for _ in range(args.gpu_num)]
    all_gather_object(to_log_list, to_log)
    if args.rank == 0:
        loss_acc = sum([v.item() for v in loss_list])/args.gpu_num
        to_log = {k: sum([v[k] for v in to_log_list])/args.gpu_num for k in to_log.keys()}
    return loss_acc, to_log


def get_epoch_range(args):
    if args.rank == 0:
        epoch_range = trange(0, args.epochs, desc="Epoch #", position=0)
    else:
        epoch_range = range(0, args.epochs)
    return epoch_range


def is_main_proc(args):
    return args.rank == 0


def call_barrier_if_distributed(args):
    if args.gpu_num > 1:
        barrier()

def get_models_if_dist(args, model, train_model, vq_model):
    if args.gpu_num > 1:
        barrier()
        model = train_model.module.model
        # vq_model = train_model.module.vq_model
    return model, None
