import argparse
import json
import os
import pickle
from typing import Tuple

import cv2
from PIL import Image

import numpy as np
import torch
from einops import rearrange
from marlin_pytorch.face_detector import FaceXZooFaceDetector
from marlin_pytorch.util import crop_with_padding
from torchvision import transforms
from os import path

from tqdm import trange

from vq_models.vq_configs import get_fs_model
from paths import DATASET_DIRS


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
    parser.add_argument('--face_crop', action='store_true', help='crop to face')
    args = parser.parse_args()

    return args

def crop_image(frame, margin=1, x=0, y=0, extra_margin=40) -> Tuple[np.ndarray, int, int, int]:
    dets = FaceXZooFaceDetector.detect_face(frame)
    old_margin = margin
    if len(dets) > 0:
        x1, y1, x2, y2, confidence = dets[0]
        # center
        x, y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        margin = int(max(abs(x2 - x1), abs(y2 - y1)) / 2)
        old_margin = margin
        margin += extra_margin
    elif x != 0 and y != 0:
        margin += extra_margin
    else:
        x, y = frame.shape[1] // 2, frame.shape[0] // 2
        margin = frame.shape[1] // 3
    # crop face
    face = crop_with_padding(frame, x - margin, x + margin, y - margin, y + margin, 0)
    face = cv2.resize(face, (frame.shape[0], frame.shape[1]))
    return face, old_margin, x, y

def crop_video(v, device):
    v = (rearrange(v, "b c t h w -> (b t) h w c").cpu().numpy() * 255).astype(np.uint8)
    face_frames = []
    margin, x, y = 1, 0, 0
    for i in range(v.shape[0]):
        # crop_face result: (H, W, C)
        face, margin, x, y = crop_image(v[i], margin, x, y)
        face_frames.append(torch.from_numpy(face))

    faces = torch.stack(face_frames)  # (T, H, W, C)
    return rearrange(faces, "(b t) h w c -> b c t h w", b=1).to(device) / 255

def crop_face(x, device):
    # x is b, t, h, w, c
    x = rearrange(x, "b t h w c -> b c t h w").to(device)

    with torch.no_grad():
        if not FaceXZooFaceDetector.inited:
            FaceXZooFaceDetector.init(
                face_sdk_path=f'{os.path.dirname(os.path.abspath(__file__))}/../facexzoo',
                device=device
            )
        x = torch.cat([crop_video(x[[i]], device) for i in range(x.shape[0])])
        return np.uint8(rearrange(x, "b c t h w -> b t h w c ").cpu().numpy() * 255)

if __name__ == '__main__':


    args = parse_args()
    root_dir = DATASET_DIRS['celebv']
    if args.face_crop:
        transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
    info = json.load(open(path.join(root_dir, 'celebvhq_info.json')))
    meta_info = info['meta_info']
    clips = info['clips']
    video_dir = path.join(root_dir, '35666')
    with open(path.join(root_dir, 'filtered_clips.pkl'), 'rb') as f:
        to_filter = pickle.load(f)
    videos = [(path.join(video_dir, f'{clip_id}.mp4'), clips[clip_id]) for clip_id in clips.keys() if clip_id not in to_filter]



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        vq_model = get_fs_model(args, device)

        def load_item(idx):
            vid_path, info = videos[idx]
            cap = cv2.VideoCapture(vid_path)
            video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_idx = range(video_len)
            video_array = []
            for i in range(frames_idx[0], frames_idx[-1] + 1):
                ret, frame = cap.read()
                if not ret:
                    raise ValueError('Error while reading frame')
                if i in frames_idx:
                    video_array.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            video_array = np.array(video_array) / 255
            video_array = crop_face(torch.from_numpy(video_array[None]).to(device), device)[0]
            # xx = (torch.from_numpy(video_array[None]) / 255) * 2 - 1
            # xx = xx.permute(0, 1, 4, 2, 3)
            # plot_grid(torch.cat([xx[:, j:j + 10] for j in range(0, min(xx.shape[1] - xx.shape[1] % 10, 100), 10)]))
            video_array = torch.stack([transform(x) for x in video_array])
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
            # # means = np.concatenate(means)
            # # stds = np.concatenate(stds)
            hs = np.concatenate(hs)
            # # return means, stds, speaker_id
            return video_array, hs, info, os.path.basename(vid_path)


        video_list = []
        if args.face_crop:
            folder_path = os.path.join(root_dir, f'all_videos_means_stds_{args.input_size}_{args.first_stage_model}_face_crop') + '/'
        else:
            folder_path = os.path.join(root_dir, f'all_videos_means_stds_{args.input_size}_{args.first_stage_model}') + '/'
        os.makedirs(folder_path, exist_ok=True)


        def save_item(i):
            video, h, info, name = load_item(i)
            np.savez(f'{folder_path}/{name}.npz', h=h, info=info)

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

