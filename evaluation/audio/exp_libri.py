import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
from tqdm.auto import tqdm
from data import get_dataset_config
from models import DiffSDAPriorKarrasTimit
from sampling import DiffusionProcess
from utils import seed_everything, generate_exp_string
import torch.utils.data
from datasets_util.LibriSpeech import LibriSpeech, libri_normalize
from HiFiGAN import mel_spectrogram, audio_path_to_data_hifi
import torchaudio
from run import parse_args
from timit_utils import voice_verification_mean
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True



def get_model_path(args):
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
    available_models = os.listdir(root)
    available_models = sorted(available_models, key=lambda x: int(x.split('-')[1].split('.')[0]))
    epoch = int(available_models[-1].split('-')[1].split('.')[0])
    path = os.path.join(root, f'model-{epoch}.pth')
    print(path)
    return path


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.r_seed)
    args.device = device = args.rank
    shape = get_dataset_config(args)
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
    spectrogram = mel_spectrogram
    args.batch_size = 1
    dataset = LibriSpeech(train=False, eval=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    model = DiffSDAPriorKarrasTimit(args, device, shape, ch_mult=args.ch_mult)
    model.requires_grad_(False)
    state_dict = torch.load(get_model_path(args))
    model.load_state_dict(state_dict)
    eer_static, eer_dynamic = voice_verification_mean(args, model, spectrogram, dataloader, resampler)
    print(f'Static EER: {eer_static} Dynamic EER: {eer_dynamic}')

