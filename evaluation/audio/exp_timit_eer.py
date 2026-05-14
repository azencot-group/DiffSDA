"""Run TIMIT speaker-verification EER (static + dynamic) on a trained
DiffSDA-Timit model. Mirrors evaluation/audio/exp_libri.py but uses the TIMIT
data loader.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import torch
import torchaudio

from data import get_dataset, get_dataset_config
from HiFiGAN import mel_spectrogram
from models import DiffSDAPriorKarrasTimit
from run import parse_args
from timit_utils import voice_verification_mean
from utils import seed_everything, generate_exp_string

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


def get_model_path(args):
    root = args.model_folder
    if args.model == 'timediffpriorkarras':
        root = os.path.join(root, 'timediffpriorkarras')
    root = os.path.join(root, generate_exp_string(args))
    available = sorted(os.listdir(root), key=lambda x: int(x.split('-')[1].split('.')[0]))
    epoch = int(available[-1].split('-')[1].split('.')[0])
    path = os.path.join(root, f'model-{epoch}.pth')
    print(path)
    return path


if __name__ == '__main__':
    import os
    args = parse_args()
    seed_everything(args.r_seed)
    args.device = device = args.rank
    shape = get_dataset_config(args)

    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
    spectrogram = mel_spectrogram

    args.batch_size = int(os.environ.get('DIFFSDA_TIMIT_EVAL_BATCH_SIZE', 64))
    _, test_loader = get_dataset(args)

    model = DiffSDAPriorKarrasTimit(args, device, shape, ch_mult=args.ch_mult)
    model.requires_grad_(False)
    state_dict = torch.load(get_model_path(args))
    model.load_state_dict(state_dict)

    eer_static, eer_dynamic = voice_verification_mean(args, model, spectrogram, test_loader, resampler)
    print(f'Static EER: {eer_static} Dynamic EER: {eer_dynamic}')
