"""Model weight paths and argument strings for evaluation.

Weights are resolved from FINAL_WEIGHTS_ROOT (see paths.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import FINAL_WEIGHTS_ROOT, MODELS_ROOT

path_to_models = {
    'vox1':      os.path.join(FINAL_WEIGHTS_ROOT, 'vox1.pth'),
    'celebv':    os.path.join(FINAL_WEIGHTS_ROOT, 'celebv.pth'),
    'mug':       os.path.join(FINAL_WEIGHTS_ROOT, 'mug.pth'),
    'mug_small': os.path.join(FINAL_WEIGHTS_ROOT, 'mug_small.pth'),
    'taichi':    os.path.join(FINAL_WEIGHTS_ROOT, 'taichi.pth'),
}

args_strings = {
    'vox1': (
        '--r_seed 42 --dataset vox1 --model timediffpriorkarras --mode eval '
        '--s_dim 512 --d_dim 12 --hidden_dim 1024 --a_dim 32 '
        '--sampling_number 4 --gpu_num 1 --video_length 10 '
        '--input_channels 4 --input_size 256 '
        '--unets_channels 192 --encoder_channels 192 '
        '--first_stage_model vq8ft --scale --latent_dataset '
        '--diffusion_steps 32 --save_epochs 20 --batch_size 8 --epochs 500 '
        f'--model_folder {FINAL_WEIGHTS_ROOT} --use_pre_split'
    ),
    'celebv': (
        '--r_seed 42 --dataset celebv --model timediffpriorkarras --mode eval '
        '--s_dim 1024 --d_dim 16 --hidden_dim 1024 --a_dim 32 '
        '--sampling_number 4 --gpu_num 1 --video_length 10 '
        '--input_channels 4 --input_size 256 '
        '--unets_channels 192 --encoder_channels 192 '
        '--first_stage_model vq8ft --scale --latent_dataset '
        '--diffusion_steps 32 --save_epochs 20 --batch_size 8 --epochs 500 '
        f'--model_folder {FINAL_WEIGHTS_ROOT} --use_pre_split'
    ),
    'mug': (
        '--r_seed 42 --dataset mug --model timediffpriorkarras --mode eval '
        '--s_dim 256 --d_dim 64 --hidden_dim 128 '
        '--sampling_number 4 --diffusion_steps 32 --epochs 50 '
        '--batch_size 4 --video_length 15 --input_size 64 --input_channels 3 '
        '--epochs_latent 500'
    ),
    'taichi': (
        '--r_seed 42 --dataset taichi --model timediffpriorkarras --mode eval '
        '--s_dim 512 --d_dim 64 --hidden_dim 1024 --a_dim 32 '
        '--sampling_number 4 --gpu_num 1 --video_length 10 '
        '--input_channels 3 --input_size 64 '
        '--unets_channels 64 --encoder_channels 64 '
        '--diffusion_steps 32 --save_epochs 20 --batch_size 8 --epochs 500 '
        f'--model_folder {FINAL_WEIGHTS_ROOT} --newsplit'
    ),
    'mug_small': (
        '--r_seed 42 --dataset mug --model timediffpriorkarras --mode eval '
        '--s_dim 128 --d_dim 16 --hidden_dim 256 '
        '--sampling_number 4 --diffusion_steps 36 --epochs 50 '
        '--batch_size 4 --video_length 15 --input_size 64 --input_channels 3'
    ),
}
