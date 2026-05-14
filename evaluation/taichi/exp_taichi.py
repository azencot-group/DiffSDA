"""TaiChi-HD reconstruction AKD/AED evaluation (paper Tab 3).

- AKD via OpenPose-style pose estimator (Cao et al. 2017) over 18 keypoints
- AED via person re-identification embedding (Hermans et al. 2017 / Layumi reid baseline)

Run from project root:
    DIFFSDA_DATASETS_ROOT=/path/to/datasets \\
    DIFFSDA_FINAL_WEIGHTS=/path/to/DiffSDA/final_weights/DiffSDA \\
    DIFFSDA_PRETRAINED_ROOT=/path/to/vq_models \\
    python -m evaluation.taichi.exp_taichi
"""

import os
import sys

# Make the project root importable when this file is run as a module/script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from reid_baseline.model import ft_net

import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from data import get_dataset_config, get_dataset
from sampling import DiffusionProcess
from run import parse_args
from vq_models.vq_configs import get_fs_model
from models import DiffSDAPriorKarras
from evaluation.models_configs import path_to_models
from pose_estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from pose_estimation.network.rtpose_vgg import get_model as get_pose_model
from pose_estimation.network.post import decode_pose


# --- pose / reid weight paths (overridable via env) ---
POSE_WEIGHTS = os.environ.get(
    'DIFFSDA_POSE_WEIGHTS',
    os.path.join(_PROJ, 'pose_estimation', 'network', 'weight', 'pose_model.pth'),
)
REID_WEIGHTS = os.environ.get(
    'DIFFSDA_REID_WEIGHTS',
    os.path.join(_PROJ, 'reid_baseline', 'reid_model.pth'),
)


# Mirrors the original TaiChi benchmark args (the upstream `--prior regular`
# flag is dropped because our run.py parser doesn't define it).
ARGS_STRING = (
    '--r_seed 42 --dataset taichi --model timediffpriorkarras --mode train '
    '--s_dim 512 --d_dim 64 --hidden_dim 1024 --a_dim 32 --mode eval '
    '--sampling_number 4 --gpu_num 1 --video_length 10 '
    '--input_channels 3 --input_size 64 --unets_channels 64 --epochs_latent 500 '
    '--encoder_channels 64 --diffusion_steps 32 --save_epochs 20 --batch_size 8 '
    '--epochs 500 --newsplit'
)


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
    return args, device, model, vq_model


def extract_joints(param, pos_model, frame_uint8):
    """Returns 18x2 keypoints; rows of [-1, -1] mark missing joints."""
    frame = frame_uint8.permute(1, 2, 0).cpu().numpy()[..., ::-1]
    multiplier = get_multiplier(frame)
    orig_paf, orig_heat = get_outputs(multiplier, frame, pos_model, 'rtpose')
    swapped_img = frame[:, ::-1, :]
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, pos_model, 'rtpose')
    paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)
    _, _to_plot, joint_list, _ = decode_pose(frame, param, heatmap, paf)
    joint_list = np.array(joint_list)
    tmp = -np.ones((18, 2))
    if len(joint_list) != 0:
        tmp[joint_list[:, -1].astype(int)] = joint_list[:, :2]
    return tmp


@torch.no_grad()
def extract_id(frame_uint8, ft_net_model, data_transforms, device):
    frame = frame_uint8.permute(1, 2, 0).cpu().numpy()
    frame = data_transforms(frame).to(device)
    return ft_net_model(frame.unsqueeze(0)).data.cpu().numpy()


@torch.no_grad()
def akd_aed(args, model, vq_model, pose_model, ft_net_model, device):
    torch.manual_seed(42)
    np.random.seed(42)
    args.shared_noise = True
    _, evalloader = get_dataset(args)
    shape = get_dataset_config(args)
    process = DiffusionProcess(args, model, device, shape)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [255., 255., 255.]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}

    scores, scores_xt = [], []
    miss_list, miss_xt = [], []
    id_scores, id_scores_xt = [], []

    for video in tqdm(evalloader):
        video = video[0]
        video_clone = video.clone()
        video = video.to(device)
        b, v, c, h, w = video.shape
        if args.first_stage_model:
            v_enc = vq_model.encode(video.view(-1, c, h, w))
            v_enc = v_enc.view(b, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
        else:
            v_enc = video
        s, d, a = model.encoder(v_enc)
        x_t = process.reverse_sampling(v_enc, a=a)
        x_rec = process.sampling(sampling_number=b, a=a)
        x_rec_with_xt = process.sampling(sampling_number=b, xT=x_t, a=a)
        if args.first_stage_model:
            x_rec = vq_model.decode(
                x_rec.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
            ).view(b, v, c, h, w)
            x_rec_with_xt = vq_model.decode(
                x_rec_with_xt.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
            ).view(b, v, c, h, w)

        x_rec = denormalize_int(torch.clip(x_rec, -1, 1)).cpu()
        x_rec_with_xt = denormalize_int(torch.clip(x_rec_with_xt, -1, 1)).cpu()
        gt = denormalize_int(video_clone)

        for i in range(b):
            for j in range(v):
                gt_kp = extract_joints(param, pose_model, gt[i, j])
                pred = extract_joints(param, pose_model, x_rec[i, j])
                pred_xt = extract_joints(param, pose_model, x_rec_with_xt[i, j])
                gt_id = extract_id(gt[i, j], ft_net_model, data_transforms, device)
                rec_id = extract_id(x_rec[i, j], ft_net_model, data_transforms, device)
                rec_id_xt = extract_id(x_rec_with_xt[i, j], ft_net_model, data_transforms, device)

                m_gt = (gt_kp[:, 0] == -1)
                m_pred = (pred[:, 0] == -1)
                m_pred_xt = (pred_xt[:, 0] == -1)
                present = np.logical_and(np.logical_not(m_gt), np.logical_not(m_pred))
                pred_fail = np.logical_and(np.logical_not(m_gt), m_pred)
                present_xt = np.logical_and(np.logical_not(m_gt), np.logical_not(m_pred_xt))
                pred_fail_xt = np.logical_and(np.logical_not(m_gt), m_pred_xt)
                present_gt = 18 - m_gt.sum()

                if present_gt != 0:
                    miss_list.append(pred_fail.sum() / present_gt)
                    miss_xt.append(pred_fail_xt.sum() / present_gt)
                if present.sum() != 0:
                    scores.append(np.mean(np.abs(gt_kp[present] - pred[present]).astype(float)))
                if present_xt.sum() != 0:
                    scores_xt.append(np.mean(np.abs(gt_kp[present_xt] - pred_xt[present_xt]).astype(float)))

                id_scores.append(np.sum(np.abs(gt_id - rec_id).astype(float) ** 2))
                id_scores_xt.append(np.sum(np.abs(gt_id - rec_id_xt).astype(float) ** 2))

        del v_enc, s, d, a, x_t, x_rec, x_rec_with_xt, gt
        torch.cuda.empty_cache()

    return {
        'AKD':         float(np.mean(scores)),
        'AKD_T':       float(np.mean(scores_xt)),
        'AED':         float(np.mean(id_scores)),
        'AED_T':       float(np.mean(id_scores_xt)),
        'AKD_missing': float(np.mean(miss_list)),
        'AKD_T_missing': float(np.mean(miss_xt)),
    }


def main():
    args, device, model, vq_model = load_diffsda_model()

    pose_model = get_pose_model('vgg19')
    pose_model.load_state_dict(torch.load(POSE_WEIGHTS))
    pose_model = torch.nn.DataParallel(pose_model).to(device)
    pose_model.float().eval()

    net = ft_net(751)
    net.load_state_dict(torch.load(REID_WEIGHTS))
    net.model.fc = nn.Sequential()
    net.classifier = nn.Sequential()
    net.to(device)

    metrics = akd_aed(args, model, vq_model, pose_model, net, device)
    print('\nTaiChi reconstruction:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

    os.makedirs('results', exist_ok=True)
    with open('results/taichi_akd_aed.txt', 'w') as f:
        f.write('  '.join(f'{k}: {v}' for k, v in metrics.items()) + '\n')


if __name__ == '__main__':
    main()
