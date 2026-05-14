import numpy as np
import os, glob
import re

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from pathlib import Path

from tqdm import tqdm

from paths import DATASET_DIRS, CLASSIFIERS_ROOT, SPLITS_ROOT

frame_name_regex = re.compile(r'([0-9]+).jpg')


def frame_number(name):
    match = re.search(frame_name_regex, str(name))
    return match.group(1)


def read_video(paths, transform):
    video = []
    for path in paths:
        f = Image.open(path)
        try:
            frame = np.asarray(f, dtype=np.float32)
            if transform:
                frame = transform(frame.astype(np.uint8))
            video.append(frame)
        finally:
            if hasattr(f, 'close'):
                f.close()

    return torch.vstack([v.unsqueeze(dim=0) for v in video])


class MugDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size=64, video_length=15, mode='train', **kwargs):
        self.root_path = Path(path)
        self.video_length = video_length
        self.extract_speed = 2
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])

        self.video_categories = list(self.root_path.glob("*"))
        self.num_labels = len(self.video_categories)
        self.subjects = []

        dataset_path = DATASET_DIRS['mug_subjects']
        # ---- create dict for one hot vector -----
        subj_dict = {}
        subject_dirs = glob.glob(os.path.join(dataset_path, '*'))
        sorted_dirs = []
        for subject_idx, subject_dir in enumerate(subject_dirs):
            sub_dirs_str = subject_dir.split('/')
            subj_idx = sub_dirs_str.index('subjects3')
            sorted_dirs.append(sub_dirs_str[subj_idx + 1])
        sorted_dirs = sorted(sorted_dirs)

        for subject_idx, sorted_dir in enumerate(sorted_dirs):
            subj_dict[sorted_dir] = subject_idx

        category2num = {
            "anger": 0,
            "disgust": 1,
            "happiness": 2,
            "fear": 3,
            "sadness": 4,
            "surprise": 5,
        }

        self.videos = []
        for category_path in self.video_categories:
            if not category_path.is_dir():
                continue

            num_categ = int(category_path.name)
            for video_path in category_path.glob("*"):
                if not video_path.is_dir():
                    continue

                video_len = len(list(video_path.glob("*.jpg")))
                if video_len >= video_length:
                    subj = 0
                    p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
                    if re.search(p, video_path.name) is not None:
                        for catch in re.finditer(p, video_path.name):
                            subj = catch[0]
                            break
                    self.videos.append((video_path, num_categ, subj_dict[subj]))
                else:
                    print(">> discarded {} (video length {} < {})\n".
                          format(video_path.parent.name, video_len, video_length))
        if mode == 'test':
            bad_samples = [514, 517, 518, 519, 520, 542, 543, 32, 36, 550, 551, 557, 562, 52, 571, 573, 67,
                           76, 628, 630, 135, 143, 659, 660, 665, 685, 177, 689, 691, 194,
                           706, 708, 720, 721, 213, 729, 745, 236, 338, 371, 404, 429,
                           434, 436, 437, 439, 440, 441, 447, 449, 461, 462, 464, 466, 483, 484, 489, 491, 494, 501,
                           510]
            self.videos = [i for j, i in enumerate(self.videos) if j not in bad_samples]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        """return video shape: (ch, frame, width, height)"""
        video_path, categ, subj = self.videos[i]

        frame_paths = np.array(sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=frame_number))

        # videos can be of various length, we randomly sample sub-sequences
        video_len = len(frame_paths)
        if video_len < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(len(frame_paths), self.video_length))
        elif video_len > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
            frame_paths = frame_paths[subsequence_idx]
        else:
            gap = video_len - self.video_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.arange(start, start + self.video_length)
            frame_paths = frame_paths[subsequence_idx]

        # read video
        video = read_video(frame_paths, self.transform)
        if len(video.shape) != 4:
            raise ValueError('invalid video shape: {}'.format(video.shape))

        # return video, i, categ, subj
        return video, categ, subj

def load_dataset(opt):
    TEST_PATH = DATASET_DIRS['mug_test']
    TRAIN_PATH = DATASET_DIRS['mug_train']
    dataset = MugDataset(TRAIN_PATH, video_length=opt.video_length, image_size=opt.input_size)
    testdataset = MugDataset(TEST_PATH, video_length=opt.video_length, mode='test', image_size=opt.input_size)
    if opt.gpu_num > 1:
        trainloder = DataLoader(dataset, batch_size=opt.batch_size, drop_last=True, num_workers=0, shuffle=False,
                                pin_memory=True, sampler=torch.utils.data.distributed.DistributedSampler(dataset))
    else:
        trainloder = DataLoader(dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4, shuffle=True, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size=opt.batch_size, drop_last=False, num_workers=0, shuffle=True, pin_memory=True)
    return trainloder, testloader



class MugDatasetPair(torch.utils.data.Dataset):
    def __init__(self, path, image_size=64, video_length=15, **kwargs):
        self.root_path = Path(path)
        self.video_length = video_length
        self.extract_speed = 2
        self.transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
        ])

        self.video_categories = list(self.root_path.glob("*"))
        self.num_labels = len(self.video_categories)
        self.subjects = []

        dataset_path = DATASET_DIRS['mug_subjects']
        # ---- create dict for one hot vector -----
        subj_dict = {}
        subject_dirs = glob.glob(os.path.join(dataset_path, '*'))
        sorted_dirs = []
        for subject_idx, subject_dir in enumerate(subject_dirs):
            sub_dirs_str = subject_dir.split('/')
            subj_idx = sub_dirs_str.index('subjects3')
            sorted_dirs.append(sub_dirs_str[subj_idx + 1])
        sorted_dirs = sorted(sorted_dirs)

        for subject_idx, sorted_dir in enumerate(sorted_dirs):
            subj_dict[sorted_dir] = subject_idx

        self.videos = []
        df_pairs = pd.read_csv(os.path.join(SPLITS_ROOT, 'mug_pairs.csv'))
        all_pairs = df_pairs.values

        # The committed CSV stores paths relative to the MUG test root
        # (e.g. mug_pre2_test/4/user014_take000_01). Rewrite the prefix to
        # the configured DATASET_DIRS['mug_test'] so the dataset works on
        # any host. Drop pairs whose folders don't have enough frames on
        # disk.
        mug_test_root = DATASET_DIRS['mug_test']
        def _remap(p):
            anchor = 'mug_pre2_test/'
            idx = p.find(anchor)
            if idx >= 0:
                return os.path.join(mug_test_root, p[idx + len(anchor):])
            return p

        valid = []
        for row in all_pairs:
            row = list(row)
            row[0] = _remap(row[0])
            row[3] = _remap(row[3])
            n_a = len(glob.glob(os.path.join(row[0], '*.jpg')))
            n_b = len(glob.glob(os.path.join(row[3], '*.jpg')))
            if n_a >= video_length and n_b >= video_length:
                valid.append(row)
        self.videos = np.array(valid, dtype=object)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        """return video shape: (ch, frame, width, height)"""
        video_path_A, _, _, video_path_B, _, _ = self.videos[i]

        frame_paths_A = np.array(sorted(glob.glob(os.path.join(video_path_A, '*.jpg')), key=frame_number))
        frame_paths_B = np.array(sorted(glob.glob(os.path.join(video_path_B, '*.jpg')), key=frame_number))

        # videos can be of various length, we randomly sample sub-sequences
        frame_paths_A = self.get_frames(frame_paths_A)
        frame_paths_B = self.get_frames(frame_paths_B)

        # read video
        video_A = read_video(frame_paths_A, self.transform)
        video_B = read_video(frame_paths_B, self.transform)
        if len(video_A.shape) != 4 or len(video_B.shape) != 4:
            raise ValueError('invalid video shape A : {} or video shape B : {}'.format(video_A.shape, video_B.shape))

        return video_A, video_B

    def get_frames(self, frame_paths):
        video_len = len(frame_paths)
        if video_len < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(len(frame_paths), self.video_length))
        elif video_len > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            start = 0
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
            frame_paths = frame_paths[subsequence_idx]
        else:
            start = 0
            subsequence_idx = np.arange(start, start + self.video_length)
            frame_paths = frame_paths[subsequence_idx]
        return frame_paths


if __name__ == '__main__':
    MUG_TEST_DATASET_PATH = DATASET_DIRS['mug_test']
    MUG_TRAIN_DATASET_PATH = DATASET_DIRS['mug_train']
    CLS_PATH = os.path.join(CLASSIFIERS_ROOT, 'mug_cls_new_contrastive.tar')
    dtest = MugDatasetPair(MUG_TEST_DATASET_PATH)
    for x, y in tqdm(dtest):
        print(x.shape, y.shape)
