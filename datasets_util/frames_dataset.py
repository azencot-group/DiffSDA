import os
import pickle

from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
from torchvision import transforms
from datasets_util.augmentation import AllAugmentationTransform

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov') or name.lower().endswith('.avi'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, video_length=15,
                 extract_speed=5, image_size=64, debug=False, new_split=False, use_pre_split=True, args=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        self.extract_speed = extract_speed
        self.args = args

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            if new_split:
                self.videos = [f'train/{v}' for v in train_videos] + [f'test/{v}' for v in test_videos]
                if use_pre_split and args is not None:
                    train_videos_basename = pickle.load(open(
                        f"{self.root_dir}/split_{args.input_size}_{args.first_stage_model}_train.pkl",
                        'rb'))
                    test_videos_basename = pickle.load(open(
                        f"{self.root_dir}/split_{args.input_size}_{args.first_stage_model}_test.pkl",
                        'rb'))
                    train_videos = [v for v in self.videos if os.path.basename(v) in train_videos_basename]
                    test_videos = [v for v in self.videos if os.path.basename(v) in test_videos_basename]
                else:
                    train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.1)
            else:
                self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos
        if debug:
            self.videos = self.videos[:100]
        self.is_train = is_train
        augmentation_params = {'flip_param': {'horizontal_flip': True, 'time_flip': True}, 'jitter_param': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.1}}
        self.all_augmentations = AllAugmentationTransform(**augmentation_params)


    def __len__(self):
        return len(self.videos)

    def get_frames_idx(self, num_frames, path):
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                .format(len(path), self.video_length))
        elif num_frames > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            extract_speed = num_frames // self.video_length
            # print(f'invalid video length: {num_frames} > {self.video_length * self.extract_speed} the new '
            #       f'extract_speed is {extract_speed}')
            needed = extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        return frames_idx

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            if self.id_sampling:
                frames = [str(s, encoding='utf-8') for s in frames]
            num_frames = len(frames)
            frame_idx = self.get_frames_idx(num_frames, path)
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            video_array = np.stack(video_array)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = self.get_frames_idx(num_frames, path)
            video_array = video_array[frame_idx]

        # if self.transform is not None:
        #     video_array = self.transform(video_array)

        # out = {}
        # if self.is_train:
        #     source = np.array(video_array[0], dtype='float32')
        #     driving = np.array(video_array[1], dtype='float32')

        #     out['driving'] = driving.transpose((2, 0, 1))
        #     out['source'] = source.transpose((2, 0, 1))
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))

        # out['name'] = video_name
        if self.args is not None and self.args.mode == 'save_latent':
            return (torch.stack([self.transform(x) for x in np.uint8(video_array*255)]), )
        elif self.is_train:
            video_array = np.array(self.all_augmentations(video_array))
        video_array = torch.stack([self.transform(x) for x in np.uint8(video_array*255)])
        
        return (video_array, )


class FramesDatasetCsvPair(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3),
                 video_length=15, extract_speed=4, image_size=64, debug=False, args=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])
        self.video_length = video_length
        self.extract_speed = extract_speed

        if os.path.exists(os.path.join(root_dir, 'celebv_pairs.csv')):
            df_pairs = pd.read_csv(os.path.join(root_dir, 'celebv_pairs.csv'))
        else:
            if os.path.exists(os.path.join(root_dir, 'train')):
                assert os.path.exists(os.path.join(root_dir, 'test'))
                print("Use predefined train-test split.")
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
                test_videos = os.listdir(os.path.join(root_dir, 'test'))
                self.videos = [f'train/{v}' for v in train_videos] + [f'test/{v}' for v in test_videos]
                train_videos_basename = pickle.load(open(
                    f"{self.root_dir}/split_{args.input_size}_{args.first_stage_model}_train.pkl",
                    'rb'))
                test_videos_basename = pickle.load(open(
                    f"{self.root_dir}/split_{args.input_size}_{args.first_stage_model}_test.pkl",
                    'rb'))
                train_videos = [v for v in self.videos if os.path.basename(v) in train_videos_basename]
                test_videos = [v for v in self.videos if os.path.basename(v) in test_videos_basename]

                self.videos = test_videos
            else:
                raise Exception("train path don't exsits.")
            df = pd.DataFrame(self.videos, columns=['filename'])
            A, B = train_test_split(df, random_state=42, test_size=0.5)
            A = A.sample(frac=1).reset_index(drop=True)
            B = B.sample(frac=1).reset_index(drop=True)
            A.columns = ['filename_A']
            B.columns = ['filename_B']
            df_pairs = pd.merge(A, B, left_index=True, right_index=True)
            df_pairs.to_csv(os.path.join(root_dir, 'celebv_pairs.csv'), index=False)
        self.videos = df_pairs.values
        if debug:
            self.videos = self.videos[:100]
        augmentation_params = {'flip_param': {'horizontal_flip': True, 'time_flip': True},
                               'jitter_param': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.1}}
        self.all_augmentations = AllAugmentationTransform(**augmentation_params)

    def __len__(self):
        return len(self.videos)

    def get_frames_idx(self, num_frames, path):
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(len(path), self.video_length))
        elif num_frames > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            extract_speed = num_frames // self.video_length
            needed = extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        return frames_idx

    def __getitem__(self, idx):
        name_A, name_B = self.videos[idx]
        path_A = os.path.join(self.root_dir, name_A)
        video_array_A = read_video(path_A, frame_shape=self.frame_shape)
        num_frames_A = len(video_array_A)
        frame_idx_A = self.get_frames_idx(num_frames_A, path_A)
        video_array_A = video_array_A[frame_idx_A]
        video_array_A = torch.stack([self.transform(x) for x in np.uint8(video_array_A * 255)])
        path_B = os.path.join(self.root_dir, name_B)
        video_array_B = read_video(path_B, frame_shape=self.frame_shape)
        num_frames_B = len(video_array_B)
        frame_idx_B = self.get_frames_idx(num_frames_B, path_B)
        video_array_B = video_array_B[frame_idx_B]
        video_array_B = torch.stack([self.transform(x) for x in np.uint8(video_array_B * 255)])
        return video_array_A, video_array_B

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}


if __name__ == "__main__":
    from tqdm import tqdm
    from utils import plot_grid
    from argparse import Namespace
    from paths import DATASET_DIRS
    args = Namespace(input_size=256, first_stage_model='vq8')
    # dataset = FramesDataset(DATASET_DIRS['taichi'], is_train=False, id_sampling=False, new_split=True, use_pre_split=True, args=args, image_size=256)
    dataset = FramesDatasetCsvPair(DATASET_DIRS['taichi'], args=args, image_size=256)
    for x, y in tqdm(dataset):
        plot_grid(x[None])
        plot_grid(y[None])