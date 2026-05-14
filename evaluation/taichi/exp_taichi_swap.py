"""TaiChi-HD conditional swap AKD/AED evaluation (paper Tab 2).

Run from project root:
    DIFFSDA_DATASETS_ROOT=/path/to/datasets \\
    DIFFSDA_FINAL_WEIGHTS=/path/to/DiffSDA/final_weights/DiffSDA \\
    DIFFSDA_PRETRAINED_ROOT=/path/to/vq_models \\
    python -m evaluation.taichi.exp_taichi_swap
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from reid_baseline.model import ft_net

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from data import get_dataset_config
from sampling import DiffusionProcess
from run import parse_args
from vq_models.vq_configs import get_fs_model
from models import DiffSDAPriorKarras
from evaluation.models_configs import path_to_models
from datasets_util.frames_dataset import FramesDatasetCsvPair
from paths import DATASET_DIRS
from pose_estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from pose_estimation.network.rtpose_vgg import get_model as get_pose_model
from pose_estimation.network.post import decode_pose


POSE_WEIGHTS = os.environ.get(
    'DIFFSDA_POSE_WEIGHTS',
    os.path.join(_PROJ, 'pose_estimation', 'network', 'weight', 'pose_model.pth'),
)
REID_WEIGHTS = os.environ.get(
    'DIFFSDA_REID_WEIGHTS',
    os.path.join(_PROJ, 'reid_baseline', 'reid_model.pth'),
)


# Mirrors the original TaiChi swap benchmark args.
ARGS_STRING = (
    '--r_seed 42 --dataset taichi --model timediffpriorkarras --mode train '
    '--s_dim 512 --d_dim 64 --hidden_dim 1024 --a_dim 32 --mode eval '
    '--sampling_number 4 --gpu_num 1 --video_length 10 '
    '--input_channels 3 --input_size 64 --unets_channels 64 --epochs_latent 500 '
    '--encoder_channels 64 --diffusion_steps 32 --save_epochs 20 --batch_size 2 '
    '--epochs 500 --newsplit'
)

_param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
_data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0., 0., 0.], [255., 255., 255.]),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def denormalize_int(data):
    return (((data + 1) / 2) * 255).to(dtype=torch.uint8)


def load_diffsda_model():
    args = parse_args(ARGS_STRING.split())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vq_model = get_fs_model(args, device) if args.first_stage_model else None
    args.input_channels = args.z_channels_vq if args.first_stage_model else args.input_channels
    shape = get_dataset_config(args)
    model = DiffSDAPriorKarras(args, device, shape, ch_mult=args.ch_mult, attn=args.attn)
    model.eval()
    model.requires_grad_(False)
    state_dict = torch.load(path_to_models['taichi'])
    model.load_state_dict(state_dict, strict=False)
    return args, device, model, vq_model, shape


@torch.no_grad()
def create_swap_images(x, y, args, model, vq_model, device, shape):
    b, v, c, h, w = x.shape
    process = DiffusionProcess(args, model, device, shape)
    x_y = torch.cat([x, y], dim=0)
    if args.first_stage_model:
        x_y = vq_model.encode(x_y.view(-1, c, args.input_size, args.input_size).to(device=device))
        x_y = x_y.view(b * 2, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
    else:
        x_y = x_y.to(device)
    s_xy, d_xy, a_xy = model.encoder(x_y)
    x_y_T = process.reverse_sampling(x_y, a_xy)
    x_y_T_swap = torch.cat([x_y_T[b:], x_y_T[:b]], dim=0)
    s_expand = s_xy.unsqueeze(1).expand(b * 2, v, args.s_dim)
    d_swap = torch.cat((d_xy[b:], d_xy[:b]), dim=0)
    a_xy_swap = torch.cat((d_swap, s_expand), dim=2)
    x_y_rec_swap = process.sampling(
        sampling_number=b * 2, xT=x_y_T_swap, a=a_xy_swap.reshape(b * 2 * v, -1)
    )
    if args.first_stage_model:
        x_y_rec_swap = vq_model.decode(
            x_y_rec_swap.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
        ).view(b * 2 * v, c, h, w).cpu()
    x_y_rec_swap = torch.clip(x_y_rec_swap.view(b * 2, v, c, h, w), -1, 1).cpu()
    return x_y_rec_swap[:b], x_y_rec_swap[b:]


def extract_joints(pose_model, frame_uint8):
    frame = frame_uint8.permute(1, 2, 0).numpy()[..., ::-1]
    multiplier = get_multiplier(frame)
    orig_paf, orig_heat = get_outputs(multiplier, frame, pose_model, 'rtpose')
    swapped_img = frame[:, ::-1, :]
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, pose_model, 'rtpose')
    paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)
    _, _to_plot, joint_list, _ = decode_pose(frame, _param, heatmap, paf)
    joint_list = np.array(joint_list)
    tmp = -np.ones((18, 2))
    if len(joint_list) != 0:
        tmp[joint_list[:, -1].astype(int)] = joint_list[:, :2]
    return tmp


@torch.no_grad()
def extract_id(frame_uint8, net, device):
    frame = frame_uint8.permute(1, 2, 0).numpy()
    frame = _data_transforms(frame).to(device)
    return net(frame.unsqueeze(0)).data.cpu().numpy()


def akd_swap(x, y, x_swap, y_swap, pose_model):
    b, v, _, _, _ = x.shape
    AKD_BaseLine, AKD_swap_subject, AKD_swap_action = [], [], []
    miss_baseline, miss_subject_swap, miss_action_swap = [], [], []
    for i in range(b):
        for j in range(v):
            gt_x = extract_joints(pose_model, x[i, j])
            gt_y = extract_joints(pose_model, y[i, j])
            pred_x = extract_joints(pose_model, x_swap[i, j])
            pred_y = extract_joints(pose_model, y_swap[i, j])
            mgx = (gt_x[:, 0] == -1)
            mgy = (gt_y[:, 0] == -1)
            mpx = (pred_x[:, 0] == -1)
            mpy = (pred_y[:, 0] == -1)
            present_x = np.logical_and(np.logical_not(mgx), np.logical_not(mpx))
            miss_x = mgx * np.logical_xor(mgx, mpx)
            present_y = np.logical_and(np.logical_not(mgy), np.logical_not(mpy))
            miss_y = mgy * np.logical_xor(mgy, mpy)
            present_xy = np.logical_and(np.logical_not(mgx), np.logical_not(mgy))
            miss_xy = np.logical_xor(mgx, mgy)
            present_xp_y = np.logical_and(np.logical_not(mpx), np.logical_not(mgy))
            miss_xp_y = mgy * np.logical_xor(mpx, mgy)
            present_x_yp = np.logical_and(np.logical_not(mgx), np.logical_not(mpy))
            miss_x_yp = mgx * np.logical_xor(mgx, mpy)
            if present_xy.sum() != 0:
                AKD_BaseLine.append(np.mean(np.abs(gt_x[present_xy] - gt_y[present_xy]).astype(float)))
                miss_baseline.append(miss_xy.sum() / 18)
            if present_xp_y.sum() != 0:
                AKD_swap_subject.append(np.mean(np.abs(pred_x[present_xp_y] - gt_y[present_xp_y]).astype(float)))
                miss_subject_swap.append(miss_xp_y.sum() / max(18 - mgy.sum(), 1))
            if present_x_yp.sum() != 0:
                AKD_swap_subject.append(np.mean(np.abs(gt_x[present_x_yp] - pred_y[present_x_yp]).astype(float)))
                miss_subject_swap.append(miss_x_yp.sum() / max(18 - mgx.sum(), 1))
            if present_x.sum() != 0:
                AKD_swap_action.append(np.mean(np.abs(gt_x[present_x] - pred_x[present_x]).astype(float)))
                miss_action_swap.append(miss_x.sum() / max(18 - mgx.sum(), 1))
            if present_y.sum() != 0:
                AKD_swap_action.append(np.mean(np.abs(gt_y[present_y] - pred_y[present_y]).astype(float)))
                miss_action_swap.append(miss_y.sum() / max(18 - mgy.sum(), 1))
    return (AKD_BaseLine, AKD_swap_subject, AKD_swap_action,
            miss_baseline, miss_subject_swap, miss_action_swap)


def aed_swap(x, y, x_swap, y_swap, net, device):
    b, v, _, _, _ = x.shape
    AED_BaseLine, AED_swap_subject, AED_swap_action = [], [], []
    for i in range(b):
        for j in range(v):
            id_x = extract_id(x[i, j], net, device)
            id_y = extract_id(y[i, j], net, device)
            id_xs = extract_id(x_swap[i, j], net, device)
            id_ys = extract_id(y_swap[i, j], net, device)
            AED_BaseLine.append(np.sum(np.abs(id_x - id_y).astype(float) ** 2))
            AED_swap_subject.append(np.sum(np.abs(id_xs - id_y).astype(float) ** 2))
            AED_swap_subject.append(np.sum(np.abs(id_x - id_ys).astype(float) ** 2))
            AED_swap_action.append(np.sum(np.abs(id_x - id_xs).astype(float) ** 2))
            AED_swap_action.append(np.sum(np.abs(id_y - id_ys).astype(float) ** 2))
    return AED_BaseLine, AED_swap_subject, AED_swap_action


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    args, device, model, vq_model, shape = load_diffsda_model()

    testset = FramesDatasetCsvPair(
        DATASET_DIRS['taichi'],
        video_length=args.video_length, image_size=args.input_size,
        debug=args.debug, extract_speed=4,
    )
    evalloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                            shuffle=False, num_workers=2)

    pose_model = get_pose_model('vgg19')
    pose_model.load_state_dict(torch.load(POSE_WEIGHTS))
    pose_model = torch.nn.DataParallel(pose_model).to(device)
    pose_model.float().eval()

    net = ft_net(751)
    net.load_state_dict(torch.load(REID_WEIGHTS))
    net.model.fc = nn.Sequential()
    net.classifier = nn.Sequential()
    net.to(device)

    AKD_BaseLine_t, AKD_swap_subject_t, AKD_swap_action_t = [], [], []
    miss_b, miss_s, miss_a = [], [], []
    AED_BaseLine_t, AED_swap_subject_t, AED_swap_action_t = [], [], []

    pbar = tqdm(evalloader)
    for x, y in pbar:
        x_swap, y_swap = create_swap_images(x, y, args, model, vq_model, device, shape)
        x_u = denormalize_int(x)
        y_u = denormalize_int(y)
        x_swap_u = denormalize_int(x_swap)
        y_swap_u = denormalize_int(y_swap)
        akd = akd_swap(x_u, y_u, x_swap_u, y_swap_u, pose_model)
        aed = aed_swap(x_u, y_u, x_swap_u, y_swap_u, net, device)
        AKD_BaseLine_t   += akd[0]
        AKD_swap_subject_t += akd[1]
        AKD_swap_action_t  += akd[2]
        miss_b += akd[3]; miss_s += akd[4]; miss_a += akd[5]
        AED_BaseLine_t   += aed[0]
        AED_swap_subject_t += aed[1]
        AED_swap_action_t  += aed[2]
        del x_swap, y_swap, x_swap_u, y_swap_u
        torch.cuda.empty_cache()

    out = {
        'AKD_BaseLine':       float(np.mean(AKD_BaseLine_t)),
        'AKD_swap_subject':   float(np.mean(AKD_swap_subject_t)),  # paper: AKD ↓ (dynamics frozen)
        'AKD_swap_action':    float(np.mean(AKD_swap_action_t)),
        'AED_BaseLine':       float(np.mean(AED_BaseLine_t)),
        'AED_swap_subject':   float(np.mean(AED_swap_subject_t)),
        'AED_swap_action':    float(np.mean(AED_swap_action_t)),  # paper: AED ↓ (static frozen)
        'missing_baseline':   float(np.mean(miss_b)),
        'missing_subject_swap': float(np.mean(miss_s)),
        'missing_action_swap':  float(np.mean(miss_a)),
    }
    print('\nTaiChi swap:')
    for k, v in out.items():
        print(f'  {k}: {v}')

    os.makedirs('results', exist_ok=True)
    with open('results/taichi_swap.txt', 'w') as f:
        f.write('  '.join(f'{k}: {v}' for k, v in out.items()) + '\n')


if __name__ == '__main__':
    main()
