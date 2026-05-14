import os
import random
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll
from torchvision.utils import make_grid
from tqdm import tqdm


def reparameterize(mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean


def gaussian_mixture(batch_size, n_dim=2, n_labels=10,
                     x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        if label >= n_labels:
            label = np.random.randint(0, n_labels)
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * math.cos(r) - y * math.sin(r)
        new_y = x * math.sin(r) + y * math.cos(r)
        new_x += shift * math.cos(r)
        new_y += shift * math.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels),
                                                     n_labels)

    return z


def swiss_roll(batch_size, noise=0.5):
    return make_swiss_roll(n_samples=batch_size, noise=noise)[0][:, [0, 2]] / 5.


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def generate_exp_string(args) -> str:
    root = f'{args.dataset}_{args.a_dim}d'
    # need to be commented out when loading old models
    if args.hidden_dim != 0:
        root += f'_{args.hidden_dim}h'
    if args.d_dim != 0:
        root += f'_{args.d_dim}dd'
    if args.s_dim != 0:
        root += f'_{args.s_dim}s'
    if args.dataset in ['timit', 'libri', 'etth', 'physionet', 'airq']:
        if args.mlp_hidden_dim != 0:
            root += f'_{args.mlp_hidden_dim}mlp_hidden_dim'
        if args.mlp_hidden_dim_enc != 0:
            root += f'_{args.mlp_hidden_dim_enc}mlp_hidden_dim_enc'
    else:
        if args.unets_channels != 0:
            root += f'_{args.unets_channels}unets_channels'
        if args.encoder_channels != 0:
            root += f'_{args.encoder_channels}encoder_channels'
        if not args.sheared_s:
            root += '_s_no_shear'
    if args.learning_rate != 0:
        root += f'_{args.learning_rate}lr'
    if args.shared_noise:
        root += '_shared_noise'
    if args.first_stage_model:
        root += '_' + args.first_stage_model
    if args.no_time:
        root += '_no_time'
    return root


def seed_everything(r_seed):
    print("Set seed: ", r_seed)
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    torch.cuda.manual_seed(r_seed)
    torch.cuda.manual_seed_all(r_seed)
    torch.backends.cudnn.deterministic = True


@torch.jit.script
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)


@torch.jit.script
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        tqdm.write('\r' + '\t'.join(entries), end='')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class LatentDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.x = torch.from_numpy(data['all_a']).float()

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


class LatentTimeDataset(Dataset):
    def __init__(self, data_path, normalize=False):
        data = np.load(data_path)
        self.s = torch.from_numpy(data['all_s']).float()
        self.d = torch.from_numpy(data['all_d']).float()
        self.normalize = normalize
        self.d_shape = self.d.shape[1:]
        if normalize:
            self.s_mean = torch.mean(self.s, dim=0)
            self.s_std = torch.std(self.s, dim=0)
            self.d_mean = torch.mean(self.d, dim=(0, 1))
            self.d_std = torch.std(self.d, dim=(0, 1))
        self.x = torch.cat([self.s, self.d.view(self.d.shape[0], -1)], dim=1)

    def _normalize(self, a):
        if self.normalize:
            s = (a[:self.s.shape[1]] - self.s_mean) / self.s_std
            d = ((a[self.s.shape[1]:].view(self.d_shape) - self.d_mean[None]) / self.d_std[None]).view(-1)
            return torch.cat([s, d], dim=0)
        return a

    def denormalize(self, a):
        if self.normalize:
            b = a.shape[0]
            s = a[:, :self.s.shape[1]] * self.s_std[None].to(a.device) + self.s_mean[None].to(a.device)
            d = (a[:, self.s.shape[1]:].view(b, *self.d_shape) * self.d_std[None].to(a.device) + self.d_mean[None].to(a.device)).view(b, -1)
            return torch.cat([s, d], dim=1)
        return a

    def __getitem__(self, index):
        return self._normalize(self.x[index])

    def __len__(self):
        return len(self.x)


def save_model(args, epoch, model):
    root = f'{args.model_folder}'
    if args.model == 'timediffpriorkarras':
        root = os.path.join(root, 'timediffpriorkarras')
    else:
        if args.model == 'vanilla':
            root = os.path.join(root, 'diff')
    root = os.path.join(root, generate_exp_string(args))
    if args.mode == "train_latent_ddim":
        root += '_latent'
        if args.latent_const:
            path = os.path.join(root, f"model-const-{args.epochs_latent}.pth")
        else:
            path = os.path.join(root, f"model{'-split' if args.latent_s_d_split else ''}-{args.epochs_latent}.pth")
    else:
        path = os.path.join(root, f'model-{epoch}.pth')
    os.makedirs(root, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch model state to {path}")


def get_model_path(args, epoch):
    root = f'{args.model_folder}'
    if args.model == 'timediffpriorkarras':
        root = os.path.join(root, 'timediffpriorkarras')
    else:
        if args.model == 'vanilla':
            root = os.path.join(root, 'diff')
    root = os.path.join(root, generate_exp_string(args))
    if args.mode == "train_latent_ddim":
        root += '_latent'
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'model-{epoch}.pth')
    return path

def denormalize(x):
    return (x + 1) / 2


vq_mean = {'celebv': np.array([-0.17344475, 0.1851073, -0.14790231, -0.16802014])}
vq_std = {'celebv': np.array([1.1577939, 0.72507054, 0.8112916, 0.8014355])}
end_path = {
    'vox1': 'VoxCeleb',
    'celebv': 'CelebV-HQ',
    'taichi': 'TAICHI/taichi-png',
}

class VQNormalizer:
    def __init__(self, args):
        from paths import DATASETS_ROOT
        root = DATASETS_ROOT

        mean_std = np.load(f'{root}/{end_path[args.dataset]}/mean_std_{args.input_size}_{args.first_stage_model}.npz')
        raw_mean = mean_std['mean']
        raw_std = np.sqrt(mean_std['variance'])

        final_mean = 0
        final_std = 0.5
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        if args.gpu_num > 1:
            self.device = args.rank
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale = torch.tensor(self.scale[None, :, None, None], device=self.device)
        self.bias = torch.tensor(self.bias[None, :, None, None], device=self.device)

    @torch.no_grad()
    def normalize(self, x):
        x = x * self.scale + self.bias
        return x

    @torch.no_grad()
    def denormalize(self, x):
        return (x - self.bias) / self.scale

    def __call__(self, x):
        return self.normalize(x)


def plot_grid(video):
    grid = (make_grid(video.view(-1, *video.shape[-3:]), nrow=video.shape[1]) + 1) / 2
    grid = grid.permute(1, 2, 0).numpy()
    sizes = np.shape(grid)
    plt_fig = plt.figure()
    plt_fig.set_size_inches((1. * sizes[1] / sizes[0]) * 10, 10, forward=False)
    ax = plt.Axes(plt_fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    plt_fig.add_axes(ax)
    ax.imshow(grid)
    plt.show()


def save_grid(video, path):
    grid = (make_grid(video.view(-1, *video.shape[-3:]), nrow=video.shape[1]) + 1) / 2
    grid = grid.permute(1, 2, 0).numpy()
    sizes = np.shape(grid)
    plt_fig = plt.figure()
    plt_fig.set_size_inches((1. * sizes[1] / sizes[0]) * 10, 10, forward=False)
    ax = plt.Axes(plt_fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    plt_fig.add_axes(ax)
    ax.imshow(grid)
    plt_fig.savefig(path)