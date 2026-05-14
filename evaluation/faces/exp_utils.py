import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import torch
import numpy as np
from evaluation.models_configs import path_to_models
from models import DiffSDAPriorKarras
from vq_models.vq_configs import get_fs_model
from data import get_dataset_config
from skimage.transform import resize
from torch.autograd import Variable


def load_models(args, dataset='vox1', path_to_model=None):
    if path_to_model is None:
        path_to_model = path_to_models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vq_model = get_fs_model(args, device) if args.first_stage_model else None
    args.input_channels = args.z_channels_vq if args.first_stage_model else args.input_channels
    shape = get_dataset_config(args)
    model = DiffSDAPriorKarras(args, device, shape, ch_mult=args.ch_mult, attn=args.attn)
    model.eval()
    model.requires_grad_(False)
    state_dict = torch.load(path_to_model[dataset])
    model.load_state_dict(state_dict, strict=False)
    return args, device, model, vq_model, shape


def seed_everything(r_seed):
    print("Set seed: ", r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    torch.cuda.manual_seed(r_seed)
    torch.cuda.manual_seed_all(r_seed)
    torch.backends.cudnn.deterministic = True


def denormalize_int(data):
    return (((data + 1) / 2) * 255).to(dtype=torch.uint8)


@torch.no_grad()
def extract_id_vec(img, net):
    id_vecs = []
    for frame in img:
        frame = denormalize_int(frame)
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


_VGG_FACE_MODEL = None


def _get_vgg_face_model():
    """Lazy-load deepface VGG-Face model (CPU). 4096-dim embeddings."""
    global _VGG_FACE_MODEL
    if _VGG_FACE_MODEL is None:
        from deepface.modules.modeling import build_model
        _VGG_FACE_MODEL = build_model(task='facial_recognition', model_name='VGG-Face')
    return _VGG_FACE_MODEL


@torch.no_grad()
def deepface_pairwise_distances(gt_imgs, rec_imgs):
    """Per-frame DeepFace.verify(distance_metric='euclidean') with default
    detector ('opencv') + alignment, matching the original AED benchmark
    protocol used in the DBSE paper.

    Returns (sum_distance, num_frames_seen, num_missing). A frame is "missing"
    when face detection fails on either gt or rec.

    gt_imgs, rec_imgs: tensors [V, C, H, W] in [-1, 1].
    """
    from deepface import DeepFace
    n = gt_imgs.shape[0]
    total = 0.0
    missing = 0
    for i in range(n):
        # Tensors are RGB; deepface (and the original calc_aed_recon.py path
        # via cv2.imread) expects BGR. Flip channels to match.
        gt_np = denormalize_int(gt_imgs[i]).permute(1, 2, 0).cpu().numpy()[..., ::-1]
        rec_np = denormalize_int(rec_imgs[i]).permute(1, 2, 0).cpu().numpy()[..., ::-1]
        # deepface 0.x expects contiguous arrays
        gt_np = np.ascontiguousarray(gt_np)
        rec_np = np.ascontiguousarray(rec_np)
        try:
            res = DeepFace.verify(
                img1_path=gt_np, img2_path=rec_np,
                model_name='VGG-Face', distance_metric='euclidean',
            )
            total += float(res['distance'])
        except Exception:
            missing += 1
    return total, n, missing



def adapt_pred_shape(preds):
    updated_preds = []
    for pred in preds:
        if isinstance(pred, list):
            pred = -1*np.ones((68, 2))
        if pred.shape[0] > 68:
            pred = pred[:68]
        updated_preds.append(pred)
    updated_preds = np.array(updated_preds)
    return updated_preds


def denormalize(data):
    return (data + 1) / 2
