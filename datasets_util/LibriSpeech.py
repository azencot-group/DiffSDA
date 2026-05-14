import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import seaborn as sns
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from paths import LIBRI_ROOT


class LibriSpeech(Dataset):

    def __init__(self, train=True, eval=False):
        self.root = LIBRI_ROOT
        self.train = train
        if train:
            self.root = os.path.join(self.root, 'train-clean-360')
        elif eval:
            self.root = os.path.join(self.root, 'test-clean')
        else:
            self.root = os.path.join(self.root, 'dev-clean')
        self.files = glob.glob(os.path.join(self.root, '**/*.flac'), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_dir, file_name = '/'.join(self.files[index].split('/')[:-1]), self.files[index].split('/')[-1][:-5]
        trans_file = [f for f in os.listdir(file_dir) if f.endswith('.txt')][0]
        with open(os.path.join(self.root, file_dir, trans_file), 'r') as f:
            lines = f.readlines()
            wrd = [' '.join(line.split(' ')[1:])[:-1] for line in lines if file_name in line][0]
        return {'wav': self.files[index], 'spk_id':self.files[index].split('/')[-3], 'wrd':wrd}
_MEAN_STD_PATH = os.path.join(LIBRI_ROOT, 'mean_std.pkl')
if os.path.exists(_MEAN_STD_PATH):
    with open(_MEAN_STD_PATH, 'rb') as f:
        mean_std = pkl.load(f)
        libri_mean_mel = torch.tensor(mean_std['mean'])
        libri_std_mel = torch.tensor(mean_std['std'])
else:
    libri_mean_mel = torch.tensor(0)
    libri_std_mel = torch.tensor(1)


def libri_normalize(args, x):
    mean = libri_mean_mel
    std = libri_std_mel
    return 0.5 * (x - mean.to(x.device)) / std.to(x.device)

def libri_denormalize(args, x):
    mean = libri_mean_mel
    std = libri_std_mel
    return 2 * x * std.to(x.device) + mean.to(x.device)


def calc_mean_and_std():
    from HiFiGAN import mel_spectrogram, audio_path_to_data_hifi
    dataset = LibriSpeech()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
    spectrogram = mel_spectrogram
    list_of_data = []
    for i, data in enumerate(tqdm(dataloader)):
        data, ch_num = audio_path_to_data_hifi(data['wav'], resampler)
        data = spectrogram(data)
        data = data.permute(0, 2, 1).numpy()
        list_of_data.append(data)
        if i == len(dataloader) // 4:
            break
    array_of_data = np.concatenate(list_of_data, axis=0)
    mean = np.mean(array_of_data, axis=(0, 1))
    std = np.std(array_of_data, axis=(0, 1))
    # save the mean and std
    with open(_MEAN_STD_PATH, 'wb') as f:
        pkl.dump({'mean': mean, 'std': std}, f)


if __name__ == '__main__':
    from HiFiGAN import mel_spectrogram, audio_path_to_data_hifi
    dataset = LibriSpeech()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
    spectrogram = mel_spectrogram
    data_iter = iter(dataloader)
    data = next(data_iter)

