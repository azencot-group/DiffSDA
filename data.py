import os

import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler

from paths import DATASET_DIRS, DATASETS_ROOT


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return torchvision.transforms.crop(img, self.x1, self.y1, self.x2 - self.x1,
                                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


class CustomTensorDataset(Dataset):
    def __init__(self, data, latents_values, latents_classes):
        self.data = data
        self.latents_values = latents_values
        self.latents_classes = latents_classes

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index]).float(),
                torch.from_numpy(self.latents_values[index]).float(),
                torch.from_numpy(self.latents_classes[index]).int())

    def __len__(self):
        return self.data.shape[0]


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


def get_dataset_config(args):
    if args.dataset in ['timit', 'libri']:
        args.mel = True if args.dataset == 'libri' else args.mel
        if args.mel:
            args.input_channels = 80
            args.unets_channels = 64
            args.encoder_channels = 64
            args.video_length = 68
            args.w_len = 320  # window length
            args.h_len = 165  # hop length
            args.power = 1    # power of the spectrogram
            args.fft_size = 400  # power of the spectrogram
        else:
            args.input_channels = 201
            args.unets_channels = 64
            args.encoder_channels = 64
            args.video_length = 20
            args.w_len = 320  # window length
            args.h_len = 165  # hop length
            args.power = 1    # power of the spectrogram
            args.fft_size = 400  # power of the spectrogram
        return (args.video_length, args.input_channels)
    elif args.dataset == 'physionet':
        args.video_length = 80
        args.input_channels = 10
        args.window_size = 4
        args.n_classes = 4
        return (args.video_length, args.input_channels)
    elif args.dataset == 'etth':
        args.video_length = 672
        args.input_channels = 6
        args.n_classes = 4
        args.window_size = 10
        return (args.video_length, args.input_channels)
    elif args.dataset == 'airq':
        args.video_length = 672
        args.input_channels = 10
        args.n_classes = 12
        args.window_size = 24
        return (args.video_length, args.input_channels)
    elif args.first_stage_model:
        shape = (args.video_length, args.input_channels, args.vq_input_size, args.vq_input_size)
    else:
        shape = (args.video_length, args.input_channels, args.input_size, args.input_size)

    return shape


def get_dataset(args):
    if args.dataset == 'mug':
        return get_mug(args)
    elif args.dataset == 'taichi':
        return get_taichi(args)
    elif args.dataset == 'vox1':
        return get_vox1(args)
    elif args.dataset == 'celebv':
        return get_celebv(args)
    elif args.dataset == 'timit':
        return get_timit(args)
    elif args.dataset == 'libri':
        return get_librispeech(args)
    elif args.dataset == 'physionet':
        return get_physionet(args)
    elif args.dataset == 'etth':
        return get_etth(args)
    elif args.dataset == 'airq':
        return get_airq(args)
    else:
        raise ValueError('Invalid dataset')




def get_mug(args):
    from datasets_util.mug_data_class import load_dataset
    return load_dataset(args)

def get_taichi(args):
    from datasets_util.frames_dataset import FramesDataset, DatasetRepeater
    taichi_root = DATASET_DIRS['taichi']
    train_dataset = FramesDataset(taichi_root, is_train=True,
                                  video_length=args.video_length, image_size=args.input_size, debug=args.debug,
                                  extract_speed=4, id_sampling=not args.newsplit,
                                  new_split=args.newsplit, use_pre_split=args.use_pre_split, args=args)
    if args.mode == 'save_latent':
        train_dataset = DatasetRepeater(train_dataset, 5)
    else:
        train_dataset = DatasetRepeater(train_dataset, 150)
    test_dataset = FramesDataset(taichi_root, is_train=False,
                                 video_length=args.video_length, image_size=args.input_size, debug=args.debug,
                                 extract_speed=4, new_split=args.newsplit, use_pre_split=args.use_pre_split, args=args)

    if args.gpu_num > 1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=6,
                                                   sampler=DistributedSampler(train_dataset),
                                                   timeout=3600)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=4)
    return train_loader, test_loader


def get_vox1(args):
    from datasets_util.voxcelebe_dataset import VoxCelebOneJpg
    data_path = DATASET_DIRS['vox1']
    train_dataset = VoxCelebOneJpg(data_path, is_train=True,
                                   video_length=args.video_length, image_size=args.input_size, args=args, use_original_split=args.use_pre_split)
    test_dataset = VoxCelebOneJpg(data_path, is_train=False,
                                  video_length=args.video_length, image_size=args.input_size, args=args, use_original_split=args.use_pre_split)

    if args.gpu_num > 1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=6,
                                                   sampler=DistributedSampler(train_dataset),
                                                   timeout=3600)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=4)
    return train_loader, test_loader


def get_latent(args):
    end_path = {
        'vox1': 'VoxCeleb',
        'celebv': 'CelebV-HQ',
        'taichi': 'TAICHI/taichi-png',
    }
    assert args.dataset in end_path, f'Invalid dataset: {args.dataset} for latent dataset.'
    root = os.path.join(DATASETS_ROOT, end_path[args.dataset])
    from datasets_util.latent_dataset import VQLatent

    if args.extract_speed == -1:
        extract_speed = 5 if args.dataset in ['taichi'] else 2
    else:
        extract_speed = args.extract_speed
    train_dataset = VQLatent(root, image_size=args.input_size, is_train=True,
                             video_length=args.video_length, first_stage_model=args.first_stage_model,
                             args=args, scale=args.scale, split=False if args.dataset in ['taichi'] else True,
                             extract_speed=extract_speed,
                             use_pre_split=args.use_pre_split)

    if args.gpu_num > 1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=6,
                                                   sampler=DistributedSampler(train_dataset))
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   num_workers=4)

    return train_loader


def get_celebv(args):
    from datasets_util.celebv_hq import CelebvHq
    if args.extract_speed == -1:
        extract_speed = 2
    else:
        extract_speed = args.extract_speed
    celebv_root = DATASET_DIRS['celebv']
    train_dataset = CelebvHq(celebv_root, is_train=True, video_length=args.video_length,
                             image_size=args.input_size, debug=args.debug,
                             use_pre_split=args.use_pre_split, extract_speed=extract_speed)
    test_dataset = CelebvHq(celebv_root, is_train=False, video_length=args.video_length,
                            image_size=args.input_size, debug=args.debug,
                            use_pre_split=args.use_pre_split, extract_speed=extract_speed)
    if args.gpu_num > 1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=6,
                                                   sampler=DistributedSampler(train_dataset),
                                                   timeout=3600)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=4)
    return train_loader, test_loader


def get_classifier(args):
    if args.dataset in ['mug']:
        from mug_cls.mug_cls_model import get_classifier
        return get_classifier()
    else:
        raise ValueError('Invalid dataset')



def get_timit(args):
    from timit_utils import prepare_timit, dataio_prep
    from speechbrain.utils.distributed import run_on_main

    # set configurations to enable data loading
    args.dataset_path = DATASET_DIRS['timit']
    args.data_folder = args.dataset_path
    annotations_root = DATASET_DIRS['timit_annotations']
    args.train_annotation = os.path.join(annotations_root, 'train.json')
    args.valid_annotation = os.path.join(annotations_root, 'valid.json')
    args.test_annotation = os.path.join(annotations_root, 'test.json')

    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": args.dataset_path,
            "save_json_train": args.train_annotation,
            "save_json_valid": args.valid_annotation,
            "save_json_test": args.test_annotation,
            "skip_prep": False,
            "uppercase": True,
        },
    )

    train_data, valid_data, test_data, label_encoder = dataio_prep(args)

    # join train and valid
    train_data.data.update(valid_data.data)
    repeats = 20
    train_list = list(train_data.data.values()) * repeats
    if args.gpu_num > 1:
        train_loader = torch.utils.data.DataLoader(train_list,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=6,
                                                   sampler=DistributedSampler(train_list),
                                                   timeout=3600)
    else:
        train_loader = DataLoader(train_list,
                                  num_workers=4,
                                  batch_size=args.batch_size,  # 128
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)
    test_loader = DataLoader(list(test_data.data.values()),
                             num_workers=4,
                             batch_size=args.batch_size,  # 128
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)

    return train_loader, test_loader

def get_librispeech(args):
    from datasets_util.LibriSpeech import LibriSpeech
    train_dataset = LibriSpeech(train=True)
    test_dataset = LibriSpeech(train=False)
    if args.gpu_num > 1:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=False,
                                                   num_workers=6,
                                                   sampler=DistributedSampler(train_dataset),
                                                   timeout=3600)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                drop_last=True,
                                                shuffle=True,
                                                num_workers=4)
    return train_loader, test_loader



def get_physionet(args):
    from datasets_util.timeseries import physionet_data_loader
    trainset, validset, testset, _ = physionet_data_loader(window_size=args.window_size, frame_ind=0,
                                                           normalize="mean_zero")
    return trainset, validset, testset


def get_etth(args):
    from datasets_util.timeseries import etth_data_loader
    trainset, validset, testset, _ = etth_data_loader(window_size=args.window_size, frame_ind=14,
                                                      normalize="mean_zero")
    return trainset, validset, testset

def get_airq(args):
    from datasets_util.timeseries import airq_data_loader
    trainset, validset, testset, _ = airq_data_loader(window_size=args.window_size, frame_ind=24,
                                                      normalize="mean_zero")
    return trainset, validset, testset