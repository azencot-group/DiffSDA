import numpy as np
from torch.utils.data import Dataset


class VoxCelebLatent(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - folder with all frames
    """

    def __init__(self, root_dir, input_size, is_train=True,
                 random_seed=42, video_length=10,
                 extract_speed=1, image_size=64, add_id=False, args=None):
        latent_path = {
            '128': 'means_stds_128.npz',
            '256': 'means_stds.npz'
        }

        latent_mean_std = {
            '128': {
                'mean': [1.4836854, -3.553759, -0.49920714, 2.0791464],
                'std': [4.74166327, 5.37248491, 4.18319351, 3.52274161]
            },
        }
        self.root_dir = root_dir
        self.data_dir = f'{root_dir}/datasets/VoxCeleb/{latent_path[str(input_size)]}'
        latent_samples = np.load(self.data_dir)
        self.mean = latent_samples['means'].astype(np.float32)
        self.std = latent_samples['stds'].astype(np.float32)

        final_mean = 0
        final_std = 0.5
        self.scale = np.float32(final_std) / np.float32(latent_mean_std[str(input_size)]['std'])
        self.bias = np.float32(final_mean) - np.float32(latent_mean_std[str(input_size)]['mean']) * self.scale

    def __len__(self):
        return len(self.mean)

    def __getitem__(self, idx):
        sampled_video = self.mean[idx] + self.std[idx] * np.random.randn(*self.mean[idx].shape[1:]).astype(np.float32)
        sampled_video = sampled_video * self.scale[None, :, None, None] + self.bias[None, :, None, None]
        return (sampled_video,)


if __name__ == "__main__":
    # from paths import DATASET_DIRS; dataset = VoxCelebOneJpg(DATASET_DIRS['vox1'], is_train=True)

    pass
    # for d in dataset:
    #     d = d[0]
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


