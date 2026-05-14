import os
import pickle
from typing import List

from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import Dataset
import glob

from tqdm import tqdm
from multiprocessing import Pool

from paths import DATASET_DIRS


def sum_sum_square_and_count(s):
    arr = np.load(s)['h']
    return arr.mean((2, 3)).sum(0), (arr ** 2).mean((2, 3)).sum(0), arr.shape[0]


def video_len(s):
    arr = np.load(s)['h']
    return s, arr.shape[0]


def speaker_id(s):
    sid = np.load(s)['speaker_id']
    return s, sid


class VQLatent(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """

    def __init__(self, root_dir, is_train=True,
                 random_seed=42, video_length=10,
                 extract_speed=2, image_size=256,
                 first_stage_model=None, args=None, scale=True, split=True, add_labels=False, use_pre_split=False):
        self.add_labels = add_labels
        self.to_scale = scale
        self.root_dir = root_dir
        self.video_length = video_length
        self.extract_speed = extract_speed
        self.first_stage_model = args.first_stage_model if first_stage_model is None else first_stage_model
        self.input_size = image_size
        if args is not None and args.dataset == 'taichi' and use_pre_split:
            data_path = f"{root_dir}/all_videos_means_stds_train_with_name_{image_size}_{self.first_stage_model}/"
        elif args is not None and args.dataset == 'celebv':
            data_path = f'{root_dir}/all_videos_means_stds_{image_size}_{self.first_stage_model}_face_crop/'
        else:
            data_path = f'{root_dir}/all_videos_means_stds_{image_size}_{self.first_stage_model}/'
        self.videos = glob.glob(os.path.join(data_path, '*.npz'))
        if args is not None and args.dataset == 'taichi' and use_pre_split:
            test_data_path = f"{root_dir}/all_videos_means_stds_test_with_name_{image_size}_{self.first_stage_model}/"
            test_videos = glob.glob(os.path.join(test_data_path, '*.npz'))
            self.videos += test_videos

        self.is_train = is_train
        # use pre-split for vox1 only
        if args is not None and args.dataset == 'celebv' and use_pre_split:
            if is_train:
                train_video = pickle.load(open(f"{self.root_dir}/train_split.pkl", 'rb'))
                self.videos = [v for v in self.videos if v.split('/')[-1][:-4] in train_video]
            else:
                test_video = pickle.load(open(f"{self.root_dir}/test_split.pkl", 'rb'))
                self.videos = [v for v in self.videos if v.split('/')[-1][:-4] in test_video]
        elif use_pre_split:
            videos = pickle.load(open(f"{self.root_dir}/split_{self.input_size}_{self.first_stage_model}_{'train' if self.is_train else 'test'}.pkl", 'rb'))
            if args is not None and args.dataset == 'taichi':
                self.videos = [v for v in self.videos if v.split('/')[-1][:-4] in videos]
            else:
                self.videos = videos
        elif split:
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

            if is_train:
                self.videos = train_videos
            else:
                self.videos = test_videos

        if not os.path.exists(f'{root_dir}/mean_std_{image_size}_{self.first_stage_model}.npz') and is_train:
            self.compute_mean_std_for_dataset()
        mean_std = np.load(f'{root_dir}/mean_std_{image_size}_{self.first_stage_model}.npz')
        self.mean = mean_std['mean']
        self.std = np.sqrt(mean_std['variance'])
        final_mean = 0
        final_std = 0.5
        self.scale = np.float32(final_std) / np.float32(self.std)
        self.bias = np.float32(final_mean) - np.float32(self.mean) * self.scale
        self.scale = self.scale[None, :, None, None]
        self.bias = self.bias[None, :, None, None]


        if args is not None and args.dataset in ['celebv', 'taichi']:
            if os.path.exists(
                    f"{self.root_dir}/filter_video_{self.input_size}_{self.first_stage_model}_{'train' if self.is_train else 'test'}.pkl"):
                self.videos = pickle.load(open(
                    f"{self.root_dir}/filter_video_{self.input_size}_{self.first_stage_model}_{'train' if self.is_train else 'test'}.pkl",
                    'rb'))
            else:
                self.videos = self.filter_bad_videos()

        if args is not None and args.debug:
            self.videos = self.videos[:100]

    def __len__(self):
        return len(self.videos)

    def compute_mean_std_for_dataset(self):
        temp = np.load(self.videos[0])
        v, c, h, w = temp['h'].shape
        # fast version
        S1 = np.zeros(c)
        S2 = np.zeros(c)
        n = 0
        subset = self.videos[:40000]
        with Pool(32) as p:
            results = list(tqdm(p.imap_unordered(sum_sum_square_and_count, subset, chunksize=32), total=len(subset)))
        for S1_, S2_, n_ in results:
            S1 += S1_
            S2 += S2_
            n += n_
        mean = S1 / n
        variance = S2 / n - (S1 / n) ** 2
        std = np.sqrt(variance)
        np.savez(f'{self.root_dir}/mean_std_{self.input_size}_{self.first_stage_model}.npz', mean=mean,
                 variance=variance, std=std)
        return mean, std

    def filter_bad_videos(self):
        filter_video = []
        with Pool(32) as p:
            for s, l in tqdm(p.imap_unordered(video_len, self.videos, chunksize=32), total=len(self.videos)):
                if l >= self.video_length:
                    filter_video.append(s)
        pickle.dump(filter_video, open(
            f"{self.root_dir}/filter_video_{self.input_size}_{self.first_stage_model}_{'train' if self.is_train else 'test'}.pkl",
            'wb'))
        return filter_video

    def generate_split_for_vox(self, train=None):
        test_id_to_name = {'id10301': 'Ernie_Hudson', 'id10300': 'Ernest_Borgnine', 'id10283': 'Eli_Roth',
                           'id10302': 'Esai_Morales',
                           'id10270': 'Eartha_Kitt', 'id10287': 'Ellen_Burstyn', 'id10296': 'Eric_McCormack',
                           'id10306': 'Eva_Green',
                           'id10275': 'Eddie_McClintock', 'id10309': 'Ezra_Miller', 'id10299': 'Erin_Andrews',
                           'id10298': 'Erik_Estrada', 'id10304': 'Eugene_Levy', 'id10293': 'Eoin_Macken',
                           'id10278': 'Edward_Asner',
                           'id10297': 'Eric_Roberts', 'id10288': 'Ellen_Wong', 'id10285': 'Elisabeth_Moss',
                           'id10305': 'Eugenio_Derbez',
                           'id10279': 'Efren_Ramirez', 'id10292': 'Emraan_Hashmi', 'id10290': 'Emile_Hirsch',
                           'id10273': 'Eddie_Izzard',
                           'id10282': 'Eleanor_Tomlinson', 'id10308': 'Evanna_Lynch', 'id10289': 'Elodie_Yung',
                           'id10281': 'Elaine_Hendrix', 'id10271': 'Ed_Westwick', 'id10307': 'Eva_Longoria',
                           'id10284': 'Eli_Wallach',
                           'id10277': 'Eduardo_Noriega', 'id10303': 'Estelle_Harris', 'id10286': 'Elle_Fanning',
                           'id10295': 'Eric_Dane',
                           'id10274': 'Eddie_Kaye_Thomas', 'id10276': 'Edgar_Wright', 'id10272': 'Eddie_Griffin',
                           'id10291': 'Emily_Atack', 'id10294': 'Eric_Braeden', 'id10280': 'Elaine_Cassidy'}
        test_name_to_id = {v: k for k, v in test_id_to_name.items()}
        with open(os.path.join(DATASET_DIRS['vox1'], 'my_id_to_name.pkl'), 'rb') as f:
            my_id_to_name = pickle.load(f)

        test_videos = []
        train_videos = []
        train = train if train is not None else self.is_train
        with Pool(32) as p:
            for s, sid in tqdm(p.imap_unordered(speaker_id, self.videos, chunksize=32), total=len(self.videos)):
                name_id = my_id_to_name[sid.item()]
                if name_id in test_name_to_id:
                    test_videos.append(s)
                else:
                    train_videos.append(s)
        pickle.dump(train_videos, open(
            f"{self.root_dir}/split_{self.input_size}_{self.first_stage_model}_train.pkl",
            'wb'))
        pickle.dump(test_videos, open(
            f"{self.root_dir}/split_{self.input_size}_{self.first_stage_model}_test.pkl",
            'wb'))
        return train_videos if train else test_videos

    def get_frames_idx(self, num_frames):
        extract_speed = self.extract_speed
        if num_frames < self.video_length * self.extract_speed:
            extract_speed = num_frames // self.video_length
        if num_frames < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                             .format(num_frames, self.video_length))
        elif num_frames >= self.video_length * extract_speed:
            needed = extract_speed * (self.video_length - 1)
            gap = num_frames - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            frames_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        else:
            raise ValueError('invalid video length: {} > {}'
                             .format(num_frames, self.video_length * extract_speed))
        return frames_idx

    def __getitem__(self, idx):
        vid_path = self.videos[idx]
        data = np.load(vid_path, allow_pickle=True)
        frames = data['h']
        num_frames = len(frames)
        frames_idx = self.get_frames_idx(num_frames)
        video_array = frames[frames_idx]
        if self.to_scale:
            video_array = video_array * self.scale + self.bias
        if self.add_labels:
            info = data['info'].item()
            action_label = torch.tensor(info['attributes']['action'], dtype=torch.long).bool()
            appearance_label = torch.tensor(info['attributes']['appearance'], dtype=torch.long).bool()
            return video_array, action_label, appearance_label
        return (video_array,)


if __name__ == '__main__':
    # dataset = VQLatent(root_dir=DATASET_DIRS['celebv'], is_train=True, random_seed=42, video_length=10,
    #                    first_stage_model='vq8', split=False, add_labels=True)
    # dataset = VQLatent(root_dir=os.path.dirname(DATASET_DIRS['vox1']), is_train=True, random_seed=42,
    #                    video_length=10,
    #                    first_stage_model='vq8ft', split=False, add_labels=False, extract_speed=1, use_pre_split=True)
    from argparse import Namespace
    args = Namespace(dataset='celebv', debug=False, first_stage_model='vq8ft')
    dataset = VQLatent(root_dir=DATASET_DIRS['celebv'], is_train=True, random_seed=42,
                       video_length=10, args=args,
                       first_stage_model='vq8ft', split=False, add_labels=False, extract_speed=1, use_pre_split=True)
    # dataset.generate_split_for_vox(True)

    # dataset.filter_bad_videos()
    dataset.compute_mean_std_for_dataset()
    # m, s = dataset.compute_mean_std_for_dataset()
    # print('mean:', dataset.mean, ', std:', dataset.std)
    # from torch.utils.data import DataLoader

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=16, pin_memory=True)
    examples = []
    print(len(dataset))
    # for d in dataset:
    #     print(d[0])
    for i, d in enumerate(tqdm(dataset)):
        if i == 8000:
            break
        examples.append(d)
    examples = np.concatenate(examples)
    print(examples.mean((0, 1, 3, 4)), examples.std((0, 1, 3, 4)))
    # print(dataset.mean, dataset.std)
