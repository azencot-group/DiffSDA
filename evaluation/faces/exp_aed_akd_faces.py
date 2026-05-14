import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import numpy as np

from datasets_util.celebv_hq import CelebvHqPair
from datasets_util.mug_data_class import MugDatasetPair
from datasets_util.voxcelebe_dataset import VoxCelebOneJpgPair
from sampling import DiffusionProcess
from run import parse_args
from models import DiffSDAPriorKarras
from data import get_dataset_config, get_dataset
from skimage.transform import resize
from torch.autograd import Variable
import face_alignment
from OpenFacePytorch.loadOpenFace import prepareOpenFace
import torch.nn.functional as F
from evaluation.models_configs import args_strings, path_to_models
from evaluation.faces.exp_utils import load_models, denormalize_int, adapt_pred_shape, deepface_pairwise_distances
from paths import DATASET_DIRS, SAMPLES_ROOT


def get_pair_dataset(args, dataset, path_to_model=None):
    if dataset == 'vox1':
        data_path = DATASET_DIRS['vox1']
        dataset = VoxCelebOneJpgPair(data_path, video_length=args.video_length, image_size=args.input_size)
    elif dataset == 'celebv':
        dataset = CelebvHqPair(DATASET_DIRS['celebv'], image_size=args.input_size,
                               video_length=args.video_length)
    elif dataset == 'mug':
       dataset = MugDatasetPair(DATASET_DIRS['mug_test'],
                                image_size=args.input_size, video_length=args.video_length)
    else:
        raise NotImplementedError(f'dataset is not implemented {dataset}')
    evalloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            drop_last=True,
                            shuffle=False,
                            num_workers=4)
    return evalloader


@torch.no_grad()
def extract_id_vec(img, net):
    id_vecs = []
    for frame in img:
        frame = frame.permute(1, 2, 0).numpy()[..., ::-1]  # RGB to BGR
        frame = resize(frame, (96, 96))  # resize to 96x96 and normalize to 0-1
        frame = np.transpose(frame, (2, 0, 1))
        with torch.no_grad():
            frame = Variable(torch.Tensor(frame)).cuda()
            frame = frame.unsqueeze(0)
            id_vec = net(frame)[0].data.cpu().numpy()
            id_vecs.append(id_vec)
    id_vecs = np.array(id_vecs)
    return id_vecs


def reconstruct(img, model, vq_model, args, shape, device):
    b, v, c, h, w = img.shape
    process = DiffusionProcess(args, model, device, shape)
    if args.first_stage_model:
        img_enc = vq_model.encode(img.view(-1, c, h, w).to(device=device)).cpu()
        img_enc = img_enc.view(b, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
    else:
        img_enc = img

    img_enc = img_enc.to(device)
    s, d, a = model.encoder(img_enc)
    xT = process.reverse_sampling(img_enc, a)
    x_rec = process.sampling(sampling_number=b, a=a)
    x_rec_xt = process.sampling(sampling_number=b, xT=xT, a=a)
    if args.first_stage_model:
        x_rec_xt = vq_model.decode(
            x_rec_xt.reshape(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).reshape(
            b * v, c, h, w).cpu()
        x_rec = vq_model.decode(x_rec.reshape(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).reshape(
            b * v,
            c, h,
            w).cpu()

    x_rec = torch.clip(x_rec.view(b, v, c, h, w), min=-1, max=1).cpu()
    x_rec_xt = torch.clip(x_rec_xt.view(b, v, c, h, w), min=-1, max=1).cpu()
    return x_rec, x_rec_xt

@torch.no_grad()
def akd_aed(args, model, vq_model, fa, device, idnet, target='mug', source=None):
    """source: source dataset name. target: target dataset name. When source == target
    (within-dataset eval) we skip the source->target resolution upsample."""
    torch.manual_seed(42)
    np.random.seed(42)
    cross_resolution = (target == 'mug' and source is not None and source != 'mug')
    _, evalloader = get_dataset(args)
    shape = get_dataset_config(args)

    AED_total_score = 0
    AED_T_total_score = 0
    AED_count = 0
    AED_count_T = 0
    AED_missing = 0
    AED_missing_T = 0
    AKD_total_score = 0
    AKD_T_total_score = 0
    AKD_count = 0
    AKD_count_T = 0
    missing = 0
    missing_T = 0
    total_good_frames = 0
    total_good_frames_T = 0

    for i, video in enumerate(tqdm(evalloader)):
        video = video[0]
        video_clone = video.clone()
        b, v, c, h, w = video.shape
        video = video.to(device)

        if cross_resolution:
            video = video.view(b * v, c, h, w)
            video = F.interpolate(video, (256, 256)).view(b, v, c, 256, 256)
        x_rec, x_rec_T = reconstruct(video, model, vq_model, args, shape, device)
        b, v, c, h, w = x_rec.shape
        if cross_resolution:
            x_rec = F.interpolate(x_rec.view(b * v, c, h, w), (64, 64)).view(b, v, c, 64, 64)
            x_rec_T = F.interpolate(x_rec_T.view(b * v, c, h, w), (64, 64)).view(b, v, c, 64, 64)
        b, v, c, h, w = video_clone.shape
        x_rec = x_rec.cpu()
        x_rec_T = x_rec_T.cpu()
        x_rec  = x_rec.view(b, v, c, h, w)
        x_rec_T  = x_rec_T.view(b, v, c, h, w)
        _, _, _, h, w = x_rec.shape
        x_rec = denormalize_int(x_rec)
        x_rec_T = denormalize_int(x_rec_T)
        video_clone = denormalize_int(video_clone)
        b, v, c, h, w = x_rec.shape


        action_gt = fa.get_landmarks_from_batch(video_clone.view(b * v, c, h, w).cpu())
        action_gt = adapt_pred_shape(action_gt)
        action_rec = fa.get_landmarks_from_batch(x_rec.view(b * v, c, h, w).cpu())
        action_rec = adapt_pred_shape(action_rec)
        action_rec_T = fa.get_landmarks_from_batch(x_rec_T.view(b * v, c, h, w).cpu())
        action_rec_T = adapt_pred_shape(action_rec_T)

        # AED via DeepFace.verify(euclidean) with default opencv detector + align,
        # matching the DBSE benchmark protocol exactly.
        gt_for_aed = video_clone.view(b * v, c, h, w).float() / 127.5 - 1
        rec_for_aed = x_rec.view(b * v, c, h, w).float() / 127.5 - 1
        rec_T_for_aed = x_rec_T.view(b * v, c, h, w).float() / 127.5 - 1
        sum_d, n_seen, n_miss = deepface_pairwise_distances(gt_for_aed, rec_for_aed)
        sum_d_T, n_seen_T, n_miss_T = deepface_pairwise_distances(gt_for_aed, rec_T_for_aed)
        AED_total_score += sum_d
        AED_T_total_score += sum_d_T
        # Original protocol divides total distance by total frames attempted
        # (missing frames contribute 0 to numerator, +1 to denominator).
        AED_count += n_seen
        AED_count_T += n_seen_T
        AED_missing += n_miss
        AED_missing_T += n_miss_T

        norms = []
        for ii in range(b * v):
            if action_gt[ii, 0, 0] == -1:
                continue
            AKD_count += 1
            total_good_frames += 1
            if action_rec[ii, 0, 0] == -1:
                missing += 1
                continue
            norms.append(np.abs(action_gt[ii] - action_rec[ii]).mean())
        AKD = np.sum(norms)
        AKD_total_score += AKD

        norms_T = []
        for ii in range(b * v):
            if action_gt[ii, 0, 0] == -1:
                continue
            AKD_count_T += 1
            total_good_frames_T += 1
            if action_rec_T[ii, 0, 0] == -1:
                missing_T += 1
                continue
            norms_T.append(np.abs(action_gt[ii] - action_rec_T[ii]).mean())
        AKD_T = np.sum(norms_T)
        AKD_T_total_score += AKD_T
    if AKD_count > 0:
        print(f'AKD     (no xT): {AKD_total_score / AKD_count}')
        akd_score = AKD_total_score / AKD_count
    else:
        print('No AKD')
        akd_score = 9999999

    if AKD_count_T > 0:
        print(f'AKD_T (with xT): {AKD_T_total_score / AKD_count_T}')
        akd_score_T = AKD_T_total_score / AKD_count_T
    else:
        print('No AKD_T')
        akd_score_T = 9999999
    aed_score = AED_total_score / max(AED_count, 1)
    aed_score_T = AED_T_total_score / max(AED_count_T, 1)
    print(f'AED     (no xT): {aed_score}  ({AED_missing} missing of {AED_count + AED_missing})')
    print(f'AED_T (with xT): {aed_score_T}  ({AED_missing_T} missing of {AED_count_T + AED_missing_T})')
    print(f'AKD missing: {missing}, AKD_T missing: {missing_T}')
    return (akd_score, missing / total_good_frames, aed_score,
            akd_score_T, missing_T / total_good_frames_T, aed_score_T)


def main(dataset='mug'):
    args = parse_args(args_strings[dataset].split())
    args, device, model, vq_model, shape = load_models(args, dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()
    AKD, missing, AED, AKD_T, missing_T, AED_T = akd_aed(args, model, vq_model, fa, device, net, target=dataset, source=dataset)
    os.makedirs('results', exist_ok=True)
    with open(f'results/{dataset}_akd_aed.txt', 'w') as f:
        f.write(
            f'AKD: {AKD}  AKD_T: {AKD_T}  '
            f'AED: {AED}  AED_T: {AED_T}  '
            f'Missing: {missing}  Missing_T: {missing_T}\n'
        )
    return AKD, missing, AED, AKD_T, missing_T, AED_T



def _build_a(s, d):
    """Mirror TimePriorEncoder.forward: a = cat(d, s_expand)."""
    b, v, _ = d.shape
    s_expand = s.unsqueeze(1).expand(b, v, s.shape[-1])
    a = torch.cat((d, s_expand), dim=2)
    return a.reshape(b * v, -1)


@torch.no_grad()
def _encode(model, vq_model, video, args):
    b, v, c, h, w = video.shape
    if args.first_stage_model:
        img_enc = vq_model.encode(video.view(-1, c, h, w)).view(
            b, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
    else:
        img_enc = video
    s, d, a = model.encoder(img_enc)
    return s, d, a, img_enc


@torch.no_grad()
def swap_akd_aed(args, model, vq_model, fa, device, target='mug', source=None):
    """Conditional swap evaluation (paper Table 2).

    For a pair (x, x_hat):
        zs swap (static frozen, dynamics from x_hat):
            sample with a = build_a(s_x, d_xhat). Compare AED with x.
        zd swap (dynamics frozen, static from x_hat):
            sample with a = build_a(s_xhat, d_x). Compare AKD with x.
    Both use xT extracted from x to keep the noise consistent (xT-conditioned).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    evalloader = get_pair_dataset(args, target)
    shape = get_dataset_config(args)
    process = DiffusionProcess(args, model, device, shape)

    AED_total = 0.0
    AKD_total = 0.0
    AED_count = 0
    AED_missing = 0
    AKD_count = 0
    missing = 0

    for x, y in tqdm(evalloader):
        x = x.to(device)
        y = y.to(device)
        bx, v, c, h, w = x.shape
        s_x, d_x, a_x, x_enc = _encode(model, vq_model, x, args)
        s_y, d_y, a_y, _ = _encode(model, vq_model, y, args)

        a_static = _build_a(s_x, d_y)   # static frozen from x, dynamics from y
        a_dynamic = _build_a(s_y, d_x)  # dynamics frozen from x, static from y

        xT_x = process.reverse_sampling(x_enc, a_x)

        x_static = process.sampling(sampling_number=bx, xT=xT_x, a=a_static)
        x_dynamic = process.sampling(sampling_number=bx, xT=xT_x, a=a_dynamic)

        if args.first_stage_model:
            x_static = vq_model.decode(
                x_static.reshape(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
            ).reshape(bx, v, c, h, w)
            x_dynamic = vq_model.decode(
                x_dynamic.reshape(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
            ).reshape(bx, v, c, h, w)

        x_static = torch.clip(x_static.view(bx, v, c, h, w), -1, 1).cpu()
        x_dynamic = torch.clip(x_dynamic.view(bx, v, c, h, w), -1, 1).cpu()
        x_clone = x.cpu()

        x_clone_u = denormalize_int(x_clone)
        x_static_u = denormalize_int(x_static)
        x_dynamic_u = denormalize_int(x_dynamic)

        # AED: identity should be preserved between x and x_static.
        # DeepFace.verify with default opencv detector + alignment, matching
        # the DBSE swap-AED benchmark protocol.
        gt_for_aed = x_clone.view(bx * v, c, h, w)
        sw_for_aed = x_static.view(bx * v, c, h, w)
        sum_d, n_seen, n_miss = deepface_pairwise_distances(gt_for_aed, sw_for_aed)
        AED_total += sum_d
        AED_count += n_seen
        AED_missing += n_miss

        # AKD: dynamics should be preserved between x and x_dynamic
        action_gt = adapt_pred_shape(fa.get_landmarks_from_batch(x_clone_u.view(bx * v, c, h, w).cpu()))
        action_sw = adapt_pred_shape(fa.get_landmarks_from_batch(x_dynamic_u.view(bx * v, c, h, w).cpu()))
        norms = []
        for ii in range(bx * v):
            if action_gt[ii, 0, 0] == -1:
                continue
            if action_sw[ii, 0, 0] == -1:
                missing += 1
                continue
            norms.append(np.abs(action_gt[ii] - action_sw[ii]).mean())
            AKD_count += 1
        AKD_total += np.sum(norms)

    AKD_score = AKD_total / max(AKD_count, 1)
    AED_score = AED_total / max(AED_count, 1)
    print(f'SWAP AED (static frozen) : {AED_score}  ({AED_missing} missing of {AED_count + AED_missing})')
    print(f'SWAP AKD (dynamics frozen): {AKD_score}  ({missing} AKD missing)')
    return AED_score, AKD_score, missing, AED_missing


def main_swap(dataset='mug', batch_size=4):
    args = parse_args(args_strings[dataset].split())
    args, device, model, vq_model, shape = load_models(args, dataset)
    args.batch_size = batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    AED, AKD, missing_akd, missing_aed = swap_akd_aed(args, model, vq_model, fa, device, target=dataset, source=dataset)
    os.makedirs('results', exist_ok=True)
    with open(f'results/{dataset}_swap.txt', 'w') as f:
        f.write(f'AED (static frozen): {AED}  AKD (dynamics frozen): {AKD}  '
                f'AKD_missing: {missing_akd}  AED_missing: {missing_aed}\n')
    return AED, AKD, missing_akd, missing_aed


def zero_shot(source='vox1', target='mug', batch_size=4):
    dataset = source
    args = parse_args(args_strings[source].split())
    args, device, model, vq_model, shape = load_models(args, dataset)
    dataset = target
    args.dataset = dataset
    args.batch_size = batch_size
    if target == 'mug':
        args.video_length = 15
        args.input_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()
    AKD, missing, AED, AKD_T, missing_T, AED_T = akd_aed(args, model, vq_model, fa, device, net, target=target, source=source)
    os.makedirs('../results', exist_ok=True)
    with open(f'results/zeroshot_model_{source}_{target}_akd_aed.txt', 'w') as f:
        f.write(f'AKD: {AKD} | Missing: {missing} | AED: {AED} | AKD_T: {AKD_T} | Missing_T: {missing_T} | AED_T: {AED_T}')
    return AKD, missing, AED, AKD_T, missing_T, AED_T


def save_results(source='vox1', target='mug', batch_size=2):
    dataset = source
    source_model = source
    args = parse_args(args_strings[source_model].split())
    args, device, model, vq_model, shape = load_models(args, dataset)
    dataset = target
    args.dataset = dataset
    args.batch_size = batch_size
    if target == 'mug' and source != 'mug':
        args.video_length = 15
        args.input_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    evalloader = get_pair_dataset(args, target)
    shape = get_dataset_config(args)
    pbar = tqdm(evalloader)
    index = 0
    for i, (x, y) in enumerate(pbar):
        video = torch.cat([x, y], dim=0)
        video_clone = video.clone()
        b, v, c, h, w = video.shape
        video = video.to(device)
        if target == 'mug' and source != 'mug':
            video = video.view(b * v, c, h, w)
            video = F.interpolate(video, (256, 256)).view(b, v, c, 256, 256)
        x_rec, x_rec_T = reconstruct(video, model, vq_model, args, shape, device)
        b, v, c, h, w = x_rec_T.shape
        if target == 'mug' and source != 'mug':
            x_rec = F.interpolate(x_rec.view(b * v, c, h, w), (64, 64)).view(b, v, c, 64, 64)
            x_rec_T = F.interpolate(x_rec_T.view(b * v, c, h, w), (64, 64)).view(b, v, c, 64, 64)

        dir = os.path.join(SAMPLES_ROOT, f'recon/model_{source}_dataset_{target}/our')
        os.makedirs(dir, exist_ok=True)
        b, v, c, h, w = video_clone.shape
        for j in range(b):
            np.savez(f'{dir}/{index}.npz', video=video_clone[j], video_rec_XT=x_rec_T[j],
                     video_rec=x_rec[j])
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mug', help='source / eval dataset')
    parser.add_argument('--target', type=str, default='mug', help='target dataset for save/zero-shot modes')
    parser.add_argument('--mode', choices=['eval', 'swap', 'save', 'zeroshot'], default='eval',
                        help='eval: reconstruction AKD/AED; swap: conditional swap AKD/AED (paper Tab 2); '
                             'save: dump NPZs; zeroshot: cross-dataset eval')
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Batch size')
    args = parser.parse_args()
    if args.mode == 'eval':
        main(args.dataset)
    elif args.mode == 'swap':
        main_swap(args.dataset, args.batch_size)
    elif args.mode == 'zeroshot':
        zero_shot(source=args.dataset, target=args.target, batch_size=args.batch_size)
    else:
        save_results(args.dataset, args.target, args.batch_size)

