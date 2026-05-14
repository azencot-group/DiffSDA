import os.path
import time
from typing import List, Tuple

import pandas as pd
from einops import rearrange
from sklearn.model_selection import train_test_split
# from imageio import mimread, get_reader
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from os import path
import json
import cv2
from tqdm import tqdm
import pickle
from marlin_pytorch.face_detector import FaceXZooFaceDetector
from marlin_pytorch.util import crop_with_padding

from paths import SPLITS_ROOT

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


def filter_by_length(path, max_length_in_seconds=30, min_length_in_seconds=3):
    assert os.path.exists(path), 'File does not exist: {path}'
    assert (path.lower().endswith('.gif') or path.lower().endswith('.mp4') or path.lower().endswith('.mov'))
    v = cv2.VideoCapture(path)
    video_len = v.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = v.get(cv2.CAP_PROP_FPS)
    if video_len / frame_rate > max_length_in_seconds or video_len / frame_rate < min_length_in_seconds:
        return False
    return True


def get_video_info(path):
    assert os.path.exists(path), 'File does not exist: {path}'
    assert (path.lower().endswith('.gif') or path.lower().endswith('.mp4') or path.lower().endswith('.mov'))
    v = cv2.VideoCapture(path)
    video_len = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = v.get(cv2.CAP_PROP_FPS)
    v.release()
    return video_len, frame_rate


class CelebvHq(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """
    emotions = ["neutral", "happy", "sadness", "anger", "fear", "surprise", "contempt", "disgust"]

    def __init__(self, root_dir, is_train=True,
                 random_seed=42, video_length=10,
                 extract_speed=2, image_size=128, add_labels=False,
                 debug=False, crop_face=False,
                 use_pre_split=True, pre_crop=True):
        self.crop_face = crop_face
        self.add_labels = add_labels
        self.root_dir = root_dir
        self.extract_speed = extract_speed
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        self.videos = []
        info = json.load(open(path.join(root_dir, 'celebvhq_info.json')))
        self.meta_info = info['meta_info']
        self.clips = info['clips']
        if pre_crop:
            video_dir = path.join(root_dir, '512_face_crop')
        else:
            video_dir = path.join(root_dir, '35666')

        if use_pre_split:
            if is_train:
                train_video = pickle.load(open(f"{self.root_dir}/train_split.pkl", 'rb'))
                self.videos = [(path.join(video_dir, f'{clip_id}.mp4'), self.clips[clip_id]) for clip_id in self.clips.keys()
                           if f'{clip_id}.mp4' in train_video]
            else:
                test_video = pickle.load(open(f"{self.root_dir}/test_split.pkl", 'rb'))
                self.videos = [(path.join(video_dir, f'{clip_id}.mp4'), self.clips[clip_id]) for clip_id in self.clips.keys()
                           if f'{clip_id}.mp4' in test_video]
            with open(path.join(root_dir, f"filtered_clips_new{'' if is_train else '_test'}.pkl"), 'rb') as f:
                to_filter = pickle.load(f)
            self.videos = [vi for vi in self.videos if vi[0].split('/')[-1][:-4] not in to_filter]
        else:
            with open(path.join(root_dir, 'filtered_clips.pkl'), 'rb') as f:
                to_filter = pickle.load(f)

            self.videos = [(path.join(video_dir, f'{clip_id}.mp4'), self.clips[clip_id]) for clip_id in self.clips.keys()
                           if clip_id not in to_filter]

            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

            if is_train:
                self.videos = train_videos
            else:
                self.videos = test_videos

        if debug:
            self.videos = self.videos[:100]
        self.is_train = is_train

    def __len__(self):
        return len(self.videos)

    def filter_videos(self):
        to_filter = []
        for vid_path, info in tqdm(self.videos):
            video_len, frame_rate = get_video_info(vid_path)
            if video_len  < self.video_length:
                to_filter.append(vid_path.split('/')[-1][:-4])
        with open(path.join(self.root_dir, 'filtered_clips_new.pkl'), 'wb') as f:
            pickle.dump(to_filter, f)



    def get_frames_idx(self, num_frames, fps):
        extract_speed = self.extract_speed
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(num_frames, self.video_length))
        if num_frames < self.video_length * extract_speed:
            extract_speed = num_frames // self.video_length
            # print(
            # f'Warning: extract_speed is changed to {extract_speed} and sample fps is {fps/extract_speed} with video length {num_frames} and fps {fps} expected extract_speed {fps // self.fps}')
        if num_frames >= self.video_length * extract_speed:
            needed = extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            raise ValueError('invalid video length: {} > {}'
                             .format(num_frames, self.video_length * extract_speed))
        return frames_idx

    def read_video(self, path, device='cuda'):
        """
        Read video which can be:
          - an image of concatenated frames
          - '.mp4' and'.gif'
          - folder with videos
        """
        assert os.path.exists(path), 'File does not exist: {path}'
        assert (path.lower().endswith('.gif') or path.lower().endswith('.mp4') or path.lower().endswith('.mov'))
        cap = cv2.VideoCapture(path)
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frames_idx = self.get_frames_idx(video_len, frame_rate)
        video = []
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frames_idx[0])
        for i in range(video_len):
            ret, frame = cap.read()
            if not ret:
                raise ValueError('Error while reading frame')
            # if i in frames_idx:
            video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video = np.array(video)
        if self.crop_face:
            video = crop_face(torch.from_numpy(video[None]/255).to(device), device)[0]
        video = video[frames_idx]
        return video

    @classmethod
    def parse_emotion_label(cls, emotion_annotation: dict) -> List[int]:
        labels = [0] * 8
        if emotion_annotation["sep_flag"]:
            for emo in emotion_annotation["labels"]:
                labels[cls.emotions.index(emo["emotion"])] = 1
            return labels
        else:
            labels[cls.emotions.index(emotion_annotation["labels"])] = 1
        return labels

    def __getitem__(self, idx):
        vid_path, info = self.videos[idx]
        video_array = self.read_video(vid_path)
        video_array = torch.stack([self.transform(x) for x in video_array])
        action_label = torch.tensor(info['attributes']['action'], dtype=torch.long).bool()
        appearance_label = torch.tensor(info['attributes']['appearance'], dtype=torch.long).bool()
        emotion_info = info['attributes']['emotion']
        emotion_label = torch.tensor(self.parse_emotion_label(emotion_info), dtype=torch.long).bool()

        return (video_array, action_label, appearance_label, emotion_label) if self.add_labels else (video_array,)


class CelebvHqPair(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """

    def __init__(self, root_dir, random_seed=42, video_length=10,
                 extract_speed=1, image_size=256):
        self.root_dir = root_dir
        self.extract_speed = extract_speed
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        df_pairs = pd.read_csv(path.join(SPLITS_ROOT, 'celebv_pairs.csv'))
        self.videos = df_pairs.values
        self.root_dir = path.join(self.root_dir, '512_face_crop')

    def __len__(self):
        return len(self.videos)


    def get_frames_idx(self, num_frames, path):
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(len(path), self.video_length))
        elif num_frames >= self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            start = 0
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            extract_speed = num_frames // self.video_length
            needed = extract_speed * (self.video_length - 1)
            start = 0
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        return frames_idx

    def read_video(self, path):
        """
        Read video which can be:
          - an image of concatenated frames
          - '.mp4' and'.gif'
          - folder with videos
        """
        assert os.path.exists(path), 'File does not exist: {path}'
        assert (path.lower().endswith('.gif') or path.lower().endswith('.mp4') or path.lower().endswith('.mov'))
        cap = cv2.VideoCapture(path)
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frames_idx = self.get_frames_idx(video_len, frame_rate)
        video = []
        for i in range(video_len):
            ret, frame = cap.read()
            if not ret:
                raise ValueError('Error while reading frame')
            video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video = np.array(video)
        video = video[frames_idx]
        return video


    def __getitem__(self, idx):
        filename_A, filename_B = self.videos[idx]
        vid_path_A = path.join(self.root_dir, filename_A)
        video_array_A = self.read_video(vid_path_A)
        video_array_A = torch.stack([self.transform(x) for x in video_array_A])
        vid_path_B = path.join(self.root_dir, filename_B)
        video_array_B = self.read_video(vid_path_B)
        video_array_B = torch.stack([self.transform(x) for x in video_array_B])
        return video_array_A, video_array_B

if __name__ == "__main__":
    # set seed for reproducibility
    # from vq import VQModel, VQModelInterface, instantiate_from_config
    from utils import plot_grid

    torch.manual_seed(42)
    np.random.seed(42)

    from paths import DATASET_DIRS
    dataset = CelebvHqPair(DATASET_DIRS['celebv'], image_size=256)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
    for x, y in tqdm(dataset):
        print(x.shape, y.shape)
        # d = d[0]
        # plot_grid(d[None])


        # d = d.permute(0, 3, 2, 1).reshape(-1, 128, 3).permute(1, 0, 2).numpy() / 2 + 0.5
        # sizes = d.shape
        # fig = plt.figure()
        # fig.set_size_inches((1. * sizes[1] / sizes[0]) * 10, 10, forward=False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(d)
        # plt.show()
