import argparse
import os

from PIL import Image

import numpy as np
import torch
from torchvision import transforms

from tqdm import trange
from vq_models.vq_configs import get_fs_model
from concurrent.futures import ThreadPoolExecutor
from skimage import io, img_as_float32
from paths import DATASET_DIRS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=int, default=-1, help='section to run')
    parser.add_argument('--range', type=int, default=10, help='section to run')
    parser.add_argument('--input_size', type=int, default=256,
                        help='expected size of input')
    parser.add_argument('--first_stage_model', type=str, default='vq8', choices=['vq4', 'vq8', 'kl8', None]
                        , help='first stage model')
    parser.add_argument('--z_channels_vq', type=int, default=4, help='z_channels')
    parser.add_argument('--vq_input_size', type=int, default=32, help='quantized input size')
    parser.add_argument('--scale', action='store_true', help='scale input')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--test_set', action='store_true', help='test set')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    root_dir = DATASET_DIRS['taichi']
    transform = transforms.Compose([
        Image.fromarray,
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
    ])
    train_folder_path = os.path.join(root_dir, f'all_videos_means_stds_train_with_name_{args.input_size}_{args.first_stage_model}')
    test_folder_path = os.path.join(root_dir, f'all_videos_means_stds_test_with_name_{args.input_size}_{args.first_stage_model}')
    index = 0
    total = 0
    train_videos = os.listdir(os.path.join(root_dir, 'train'))
    test_videos = os.listdir(os.path.join(root_dir, 'test'))


    print(total)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        vq_model = get_fs_model(args, device)


        def load_item(idx, is_train=True):
            if is_train:
                name = train_videos[idx]
                path = os.path.join(root_dir, 'train', name)
            else:
                name = test_videos[idx]
                path = os.path.join(root_dir, 'test', name)
            frames = os.listdir(path)
            num_frames = len(frames)
            if num_frames == 0:
                return None
            frames_idx = range(num_frames)
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frames_idx]
            video_array = np.stack(video_array)
            video_array = torch.stack([transform(x) for x in np.uint8(video_array * 255)])
            hs = []
            start_idx = 0
            while num_frames > 0:
                current_video_array = video_array[start_idx:start_idx + 100]
                with torch.no_grad():
                    h = vq_model.encode(current_video_array.to(device)).cpu().numpy()
                hs.append(h)
                start_idx += 100
                num_frames -= 100
            h = np.concatenate(hs, axis=0)
            return h, name


        video_list = []
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)


        def save_item(i, is_train=True):
            folder_path = train_folder_path if is_train else test_folder_path
            h, name = load_item(i, is_train=is_train)
            if h is not None:
                np.savez(f'{folder_path}/{name}.npz', h=h)

        assert args.section <= args.range
        videos = train_videos if not args.test_set else test_videos
        if args.section != -1:
            video_len = len(videos)
            start = (args.section - 1) * video_len // args.range
            end = min(args.section * video_len // args.range, video_len)
        else:
            start = 0
            end = len(videos)
        print(len(videos))
        with ThreadPoolExecutor(max_workers=6) as executor:
            for i in trange(start, end):
                f = executor.submit(save_item, i, not args.test_set)
                f.result()

