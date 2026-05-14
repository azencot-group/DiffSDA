import argparse
import os

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


# from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=int, default=-1, help='section to run')
    parser.add_argument('--input_size', type=int, default=256,
                        help='expected size of input')
    parser.add_argument('--first_stage_model', type=str, default='vq8ft', choices=['vq4', 'vq8', 'kl8', 'vq8ft', None]
                        , help='first stage model')
    parser.add_argument('--z_channels_vq', type=int, default=4, help='z_channels')
    parser.add_argument('--vq_input_size', type=int, default=32, help='quantized input size')
    parser.add_argument('--scale', action='store_true', help='scale input')
    parser.add_argument('--device', type=str, default='cuda', help='device')
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
            video_len = video_array.shape[0]
            start_idx = 0
            # means = []
            # stds = []
            hs = []
            while video_len > 0:
                current_video_array = video_array[start_idx:start_idx + 100]
                # h = vq_model.encode(current_video_array.to(device))
                h = vq_model.encode(current_video_array.to(device)).cpu().numpy()
                hs.append(h)
                # mean = h.mode().detach().cpu().numpy()
                # means.append(mean)
                # std = h.std.detach().cpu().numpy()
                # stds.append(std)
                start_idx += 100
                video_len -= 100
            # means = np.concatenate(means)
            # stds = np.concatenate(stds)
            hs = np.concatenate(hs)
            # return means, stds, speaker_id
            return hs, speaker_id


        video_list = []
        folder_path = os.path.join(DATASETS_ROOT, 'VoxCeleb',
                                   f'all_videos_means_stds_{args.input_size}_{args.first_stage_model}') + '/'
        os.makedirs(folder_path, exist_ok=True)
        os.chmod(folder_path, 0o755)


        def save_item(i):
            # mean, std, speaker_id = load_item(i)
            h, speaker_id = load_item(i)
            # np.savez(f'{folder_path}/{i}.npz', mean=mean, std=std, speaker_id=speaker_id)
            np.savez(f'{folder_path}/{i}.npz', h=h, speaker_id=speaker_id)
            os.chmod(f'{folder_path}/{i}.npz', 0o777)

        # with Pool(4) as p:
        #     p.map(save_item, range(len(videos)))
        # process_map(save_item, range(len(videos)), max_workers=4, chunksize=1)
        assert args.section <= 10
        if args.section != -1:
            video_len = len(videos)
            start = (args.section-1) * video_len // 10
            end = min(args.section * video_len // 10, video_len)
        else:
            start = 0
            end = len(videos)
        for i in trange(start, end):
            save_item(i)
