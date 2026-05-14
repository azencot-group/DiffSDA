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

from utils import plot_grid
from paths import DATASET_DIRS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=int, default=-1, help='section to run')
    parser.add_argument('--input_size', type=int, default=512,
                        help='expected size of input')
    parser.add_argument('--device', type=str, default='cuda', help='device')
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
    transform = transforms.Compose([
        Image.fromarray,
        transforms.Resize(args.input_size),
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

        def load_item(idx):
            vid_path, info = videos[idx]
            cap = cv2.VideoCapture(vid_path)
            video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
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
            video_arrays = []
            for i in range(0, len(video_array), 100):
                current = video_array[i:i + 100]
                current = crop_face(torch.from_numpy(current[None]).to(device), device)[0]
                video_arrays.append(current)
            video_array = np.concatenate(video_arrays)
            return video_array, info, os.path.basename(vid_path), fps


        video_list = []
        folder_path = os.path.join(root_dir, f'{args.input_size}_face_crop') + '/'
        os.makedirs(folder_path, exist_ok=True)


        def save_item(i):
            vid_path, _ = videos[i]
            name = os.path.basename(vid_path)
            if os.path.exists(f'{folder_path}/{name}'):
                return
            video, info, name, fps = load_item(i)
            # np.savez(f'{folder_path}/{name}.npz', video=video, info=info)
            # save as mp4 file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'{folder_path}/{name}', fourcc, fps, (args.input_size, args.input_size))
            for frame in video:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()


        # with Pool(4) as p:
        #     p.map(save_item, range(len(videos)))
        # process_map(save_item, range(len(videos)), max_workers=4, chunksize=1)
        assert args.section <= 20
        if args.section != -1:
            video_len = len(videos)
            start = (args.section-1) * video_len // 20
            end = min(args.section * video_len // 20, video_len)
        else:
            start = 0
            end = len(videos)
        for i in trange(start, end):
            try:
                save_item(i)
            except Exception as e:
                print(f'Error: {e}')
                continue


