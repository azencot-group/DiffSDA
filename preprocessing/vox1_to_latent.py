import argparse
import os
import pickle

from PIL import Image
from skimage import io, img_as_float32

import numpy as np
import torch
import glob
from torchvision import transforms
from os import path

from tqdm import trange

from vq_models.vq_configs import get_fs_model
from paths import DATASET_DIRS, DATASETS_ROOT


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int, default=128,
                        help='expected size of input')
    parser.add_argument('--first_stage_model', type=str, default='kl8', choices=['vq4', 'vq8', 'kl8', None]
                        , help='first stage model')
    parser.add_argument('--z_channels_vq', type=int, default=4, help='z_channels')
    parser.add_argument('--vq_input_size', type=int, default=32, help='quantized input size')
    args = parser.parse_args()

    return args


if __name__ == '__main__':


    args = parse_args()
    root_dir = DATASET_DIRS['vox1']
    transform = transforms.Compose([
        Image.fromarray,
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
    ])
    assert os.path.exists(path.join(root_dir, 'data'))
    speaker_list = os.listdir(path.join(root_dir, 'data'))
    speaker_to_id = {speaker: i for i, speaker in enumerate(speaker_list)}
    id_to_speaker = {i: speaker for i, speaker in enumerate(speaker_list)}
    videos = []
    for speaker in speaker_list:
        for yt_id in os.listdir(path.join(root_dir, 'data', speaker, '1.6')):
            for vid_id in os.listdir(path.join(root_dir, 'data', speaker, '1.6', yt_id)):
                videos.append((path.join(root_dir, 'data', speaker, '1.6', yt_id, vid_id),
                                    speaker_to_id[speaker], yt_id, vid_id))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        vq_model = get_fs_model(args, device)

        def load_item(idx):
            vid_folder, speaker_id, _, _ = videos[idx]
            frames = sorted(glob.glob(path.join(vid_folder, '*.jpg')))
            num_frames = len(frames)
            frames_idx = range(num_frames)
            video_array = [img_as_float32(io.imread(os.path.join(vid_folder, frames[idx]))) for idx in frames_idx]
            video_array = np.stack(video_array)
            video_array = torch.stack([transform(x) for x in np.uint8(video_array * 255)])
            video_array = vq_model.encode(video_array.to(device))
            video_array_mean = video_array.mean.detach().cpu().numpy()
            video_array_std = video_array.std.detach().cpu().numpy()
            return video_array_mean, video_array_std, speaker_id


        video_list = []
        for i in trange(len(videos)):
            video_array_mean, video_array_std, speaker_id = load_item(i)
            video_list.append([video_array_mean, video_array_std, speaker_id])

        save_path = os.path.join(DATASETS_ROOT, 'VoxCeleb', f'all_videos_means_stds_{args.input_size}.pkl')
        pickle.dump(video_list, open(save_path, 'wb'))
        print(f'saved to {save_path}')






