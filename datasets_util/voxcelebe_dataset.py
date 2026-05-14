import os
import pickle

import cv2
from skimage.color import gray2rgb
from skimage import io, img_as_float32
from imageio import mimread
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
from torchvision import transforms
from os import path

from tqdm import tqdm

from paths import DATASET_DIRS, SPLITS_ROOT

test_id_to_name = {'id10301': 'Ernie_Hudson', 'id10300': 'Ernest_Borgnine', 'id10283': 'Eli_Roth', 'id10302': 'Esai_Morales',
           'id10270': 'Eartha_Kitt', 'id10287': 'Ellen_Burstyn', 'id10296': 'Eric_McCormack', 'id10306': 'Eva_Green',
           'id10275': 'Eddie_McClintock', 'id10309': 'Ezra_Miller', 'id10299': 'Erin_Andrews',
           'id10298': 'Erik_Estrada', 'id10304': 'Eugene_Levy', 'id10293': 'Eoin_Macken', 'id10278': 'Edward_Asner',
           'id10297': 'Eric_Roberts', 'id10288': 'Ellen_Wong', 'id10285': 'Elisabeth_Moss', 'id10305': 'Eugenio_Derbez',
           'id10279': 'Efren_Ramirez', 'id10292': 'Emraan_Hashmi', 'id10290': 'Emile_Hirsch', 'id10273': 'Eddie_Izzard',
           'id10282': 'Eleanor_Tomlinson', 'id10308': 'Evanna_Lynch', 'id10289': 'Elodie_Yung',
           'id10281': 'Elaine_Hendrix', 'id10271': 'Ed_Westwick', 'id10307': 'Eva_Longoria', 'id10284': 'Eli_Wallach',
           'id10277': 'Eduardo_Noriega', 'id10303': 'Estelle_Harris', 'id10286': 'Elle_Fanning', 'id10295': 'Eric_Dane',
           'id10274': 'Eddie_Kaye_Thomas', 'id10276': 'Edgar_Wright', 'id10272': 'Eddie_Griffin',
           'id10291': 'Emily_Atack', 'id10294': 'Eric_Braeden', 'id10280': 'Elaine_Cassidy'}
test_name_to_id = {v: k for k, v in test_id_to_name.items()}


class VoxCelebOneJpg(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """

    def __init__(self, root_dir, is_train=True,
                 random_seed=42, video_length=10,
                 extract_speed=1, image_size=64, add_id=False, args=None, use_original_split=False):
        self.add_id = add_id
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        self.extract_speed = extract_speed
        assert os.path.exists(path.join(root_dir, 'data'))
        self.speaker_list = os.listdir(path.join(root_dir, 'data'))
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(self.speaker_list)}
        self.id_to_speaker = {i: speaker for i, speaker in enumerate(self.speaker_list)}
        self.videos = []
        train_videos = []
        test_videos = []
        if os.path.exists(path.join(root_dir, 'video.pkl')) and not use_original_split:
            with open(path.join(root_dir, 'video.pkl'), 'rb') as f:
                self.videos = pickle.load(f)
        else:
            for speaker in self.speaker_list:
                for yt_id in os.listdir(path.join(root_dir, 'data', speaker, '1.6')):
                    for vid_id in os.listdir(path.join(root_dir, 'data', speaker, '1.6', yt_id)):
                        entry = (path.join(root_dir, 'data', speaker, '1.6', yt_id, vid_id),
                                                self.speaker_to_id[speaker], yt_id, vid_id)
                        if use_original_split:
                            if speaker in test_name_to_id:
                                test_videos.append(entry)
                            else:
                                train_videos.append(entry)
                        else:
                            self.videos.append(entry)
            if not use_original_split:
                with open(path.join(root_dir, 'video.pkl'), 'wb') as f:
                    pickle.dump(self.videos, f)

        if not use_original_split:
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if args and args.debug:
            self.videos = self.videos[:100]

    def __len__(self):
        return len(self.videos)

    def get_frames_idx(self, num_frames, path):
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(len(path), self.video_length))
        elif num_frames >= self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            raise ValueError('invalid video length: {} > {}'
                             .format(len(path), self.video_length * self.extract_speed))
        return frames_idx

    def __getitem__(self, idx):
        vid_folder, speaker_id, _, _ = self.videos[idx]
        frames = sorted(glob.glob(path.join(vid_folder, '*.jpg')))
        num_frames = len(frames)
        frames_idx = self.get_frames_idx(num_frames, frames)
        video_array = [img_as_float32(io.imread(os.path.join(vid_folder, frames[idx]))) for idx in frames_idx]
        video_array = np.stack(video_array)
        video_array = torch.stack([self.transform(x) for x in np.uint8(video_array * 255)])
        return (video_array, speaker_id) if self.add_id else (video_array,)


class VoxCelebOneJpgPair(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """

    def __init__(self, root_dir, video_length=10, extract_speed=2, image_size=256, args=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        self.extract_speed = extract_speed
        assert os.path.exists(path.join(root_dir, 'data'))
        self.speaker_list = os.listdir(path.join(root_dir, 'data'))
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(self.speaker_list)}
        self.id_to_speaker = {i: speaker for i, speaker in enumerate(self.speaker_list)}
        videos = []
        df_pairs = pd.read_csv(path.join(SPLITS_ROOT, 'vox1_pairs.csv'))
        self.videos = df_pairs.values


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

    def __getitem__(self, idx):
        vid_folder_A, _, _, _, vid_folder_B, _, _, _ = self.videos[idx]
        vid_folder_A = path.join(self.root_dir, vid_folder_A)
        vid_folder_B = path.join(self.root_dir, vid_folder_B)
        if not os.path.exists(vid_folder_A) or not os.path.exists(vid_folder_B):
            raise ValueError(f'Folder does not exist: {vid_folder_A} or {vid_folder_B}')
        frames_A = sorted(glob.glob(path.join(vid_folder_A, '*.jpg')))
        frames_B = sorted(glob.glob(path.join(vid_folder_B, '*.jpg')))
        num_frames_A = len(frames_A)
        num_frames_B = len(frames_B)
        frames_idx_A = self.get_frames_idx(num_frames_A, frames_A)
        frames_idx_B = self.get_frames_idx(num_frames_B, frames_B)
        video_array_A = [img_as_float32(io.imread(os.path.join(vid_folder_A, frames_A[idx]))) for idx in frames_idx_A]
        video_array_A = np.stack(video_array_A)
        video_array_A = torch.stack([self.transform(x) for x in np.uint8(video_array_A * 255)])
        video_array_B = [img_as_float32(io.imread(os.path.join(vid_folder_B, frames_B[idx]))) for idx in frames_idx_B]
        video_array_B = np.stack(video_array_B)
        video_array_B = torch.stack([self.transform(x) for x in np.uint8(video_array_B * 255)])
        return video_array_A, video_array_B



class VoxCelebTwoMp4(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """

    def __init__(self, root_dir,
                 video_length=10,
                 extract_speed=1, image_size=256, args=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        self.extract_speed = extract_speed
        self.videos = []
        if os.path.exists(path.join(root_dir, f'video.pkl')):
            with open(path.join(root_dir, 'video.pkl'), 'rb') as f:
                self.videos = pickle.load(f)
        else:
            mp4_dir = path.join(root_dir, 'mp4')
            id_list = os.listdir(mp4_dir)
            for person_id in id_list:
                for yt_id in os.listdir(path.join(mp4_dir, person_id)):
                    for vid_id in os.listdir(path.join(mp4_dir, person_id, yt_id)):
                        self.videos.append((path.join(mp4_dir, person_id, yt_id, vid_id), person_id, yt_id, vid_id))
            with open(path.join(root_dir, 'video.pkl'), 'wb') as f:
                pickle.dump(self.videos, f)

        if args and args.debug:
            self.videos = self.videos[:100]

    def __len__(self):
        return len(self.videos)

    def get_frames_idx(self, num_frames):
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(num_frames, self.video_length))
        elif num_frames > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            raise ValueError('invalid video length: {} > {}'
                             .format(num_frames, self.video_length * self.extract_speed))
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
        frames_idx = self.get_frames_idx(video_len)
        video = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_idx[0])
        for i in range(frames_idx[0], frames_idx[-1] + 1):
            ret, frame = cap.read()
            if not ret:
                raise ValueError('Error while reading frame')
            if i in frames_idx:
                video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video = np.array(video)
        return video

    def __getitem__(self, idx):
        vid_path, _, _, _ = self.videos[idx]
        video_array = self.read_video(vid_path)
        video_array = torch.stack([self.transform(x) for x in video_array])
        return (video_array,)


if __name__ == "__main__":
    # dataset = VoxCelebOneJpg(DATASET_DIRS['vox1'], is_train=True, use_original_split=True)
    dataset = VoxCelebOneJpgPair(DATASET_DIRS['vox1'])
    # dataset = VoxCelebTwoMp4(DATASET_DIRS['vox2_dev'])


    # from vq_configs import get_fs_model
    for x, y in tqdm(dataset):
        print(x.shape)
        print(y.shape)
#     the_list = []
#     with torch.no_grad():
#         for d in tqdm(dataloader):
#             d = d[0]
#             b, v, c, h, w = d.shape
#             d = vq_model.encode(d.to(device).view(-1, c, h, w))
#             tqdm.write(f'{d.mean((0, 2, 3))}, {d.std((0, 2, 3))}')
#             the_list.append(d.cpu().numpy())
#         the_list = np.concatenate(the_list)
#         print('mean', the_list.mean((0, 2, 3)))
#         print('std', the_list.std((0, 2, 3)))
