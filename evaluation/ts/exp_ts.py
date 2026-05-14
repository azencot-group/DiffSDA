import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
import numpy as np
import torch

from data import get_dataset, get_dataset_config
from models import DiffSDAPriorKarrasTS
from evaluation.ts.ts_eval_etth import train_predictor_and_test_avg_oil_temp
from evaluation.ts.ts_eval_utils import train_and_test_classifier
from run import parse_args
from evaluation.ts.ts_eval_utils_predictor import train_predictor_and_test_mortality
from utils import seed_everything
from paths import FINAL_WEIGHTS_ROOT

# Released time-series weights live under $DIFFSDA_FINAL_WEIGHTS/timeseries/.
# Override an individual path with the matching DIFFSDA_TS_<DATASET>_WEIGHTS env var.
weights_path = {
    'physionet': os.environ.get(
        'DIFFSDA_TS_PHYSIONET_WEIGHTS',
        os.path.join(FINAL_WEIGHTS_ROOT, 'timeseries', 'physionet.pth'),
    ),
    'airq': os.environ.get(
        'DIFFSDA_TS_AIRQ_WEIGHTS',
        os.path.join(FINAL_WEIGHTS_ROOT, 'timeseries', 'airq.pth'),
    ),
    'etth': os.environ.get(
        'DIFFSDA_TS_ETTH_WEIGHTS',
        os.path.join(FINAL_WEIGHTS_ROOT, 'timeseries', 'etth.pth'),
    ),
}


def load_model(args):
    shape = get_dataset_config(args)
    model = DiffSDAPriorKarrasTS(args, args.device, shape, ch_mult=args.ch_mult)
    state_dict = torch.load(weights_path[args.dataset])
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    return model




def eval_ts(args, dataloader, validloader, evalloader, model):
    model.eval()
    device = args.device
    test_accuracies, test_losses = train_and_test_classifier(args, dataloader, validloader, evalloader, model, device)
    if args.dataset == 'physionet':
        cv_loss = []
        cv_acc = []
        cv_auroc = []
        for cv in range(3):
            test_loss, test_acc, test_auroc = train_predictor_and_test_mortality(args, dataloader, validloader,
                                                                                 evalloader, model, device)
            cv_loss.append(test_loss)
            cv_acc.append(test_acc)
            cv_auroc.append(test_auroc)
        print('eval/cv_loss', np.mean(cv_loss), 'std', np.std(cv_loss))
        print('eval/AUPRC', np.mean(cv_acc), 'std', np.std(cv_acc))
        print('eval/AUROC', np.mean(cv_auroc), 'std', np.std(cv_auroc))
    print('eval/acc', test_accuracies)
    print('eval/accuracy', np.mean(test_accuracies)*100, 'std', np.std(test_accuracies)*100)

args_strings = {
    'physionet': '''--mode train --r_seed 42 --dataset physionet --model timediffpriorkarras --s_dim 24 --d_dim 2 --hidden_dim 96 --diffusion_steps 24 --batch_size 128 --learning_rate 5e-5 --mlp_hidden_dim 256 --mlp_hidden_dim_enc 96 --ch_mult 1 2 2 2''',
    'airq':      '''--mode train --r_seed 42 --dataset airq --model timediffpriorkarras --s_dim 16 --d_dim 4 --hidden_dim 512 --diffusion_steps 16 --batch_size 128 --learning_rate 1e-4 --mlp_hidden_dim 256 --mlp_hidden_dim_enc 128 --ch_mult 1 2 2 2''',
    'etth':      '''--mode train --r_seed 42 --dataset etth --model timediffpriorkarras --s_dim 16 --d_dim 4 --hidden_dim 512 --diffusion_steps 32 --batch_size 128 --learning_rate 1e-4 --mlp_hidden_dim 128 --mlp_hidden_dim_enc 256 --ch_mult 1 2 2 2'''
}


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser()
    cli.add_argument('--dataset', choices=list(args_strings.keys()), default='etth',
                     help='which time-series dataset to evaluate')
    cli_args = cli.parse_args()

    args_string = args_strings[cli_args.dataset]
    args = parse_args(args_string.split())
    seed_everything(args.r_seed)
    args.device = device = args.rank
    model = load_model(args)
    model = model.to(device)
    dataloader, validloader, evalloader = get_dataset(args)
    if args.dataset in ['physionet', 'airq']:
        eval_ts(args, dataloader, validloader, evalloader, model)
    else:
        test_loss = train_predictor_and_test_avg_oil_temp(args, dataloader, validloader, evalloader, model, device)
        print(test_loss)
        print('eval/MAE', np.mean(test_loss), 'std', np.std(test_loss))
