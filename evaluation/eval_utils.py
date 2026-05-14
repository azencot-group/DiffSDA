import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import copy
import os
from scipy.io.wavfile import write
import distinctipy
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.asyncio import tqdm, trange

from data import get_dataset_config, get_dataset, get_classifier, get_latent
from datasets_util.LibriSpeech import libri_denormalize, libri_normalize
from loggers import BaseLogger, TqdmLogger
from models import LatentDiff, DiffSDAPriorKarras, LatentDiffSplit, LatentDiffConst
from mug_cls.mug_cls_model import classifier_MUG
from sampling import LatentDiffusionProcess, DiffusionProcess
from timit_utils import timit_denormalize
from utils import generate_exp_string, LatentTimeDataset, LatentDataset, seed_everything, AverageMeter, ProgressMeter, \
    GradualWarmupScheduler, save_model, denormalize
import torchaudio
from HiFiGAN import mel_spectrogram, audio_path_to_data_hifi
from timit_utils import timit_normalize, audio_path_to_data

def sample_fix(args, logger, model, model2, shape, device, num_samples=5, vq_model=None):
    _, evalloder = get_dataset(args)
    process_latent = LatentDiffusionProcess(args, model2, device)
    for i, data in enumerate(evalloder):
        if i == num_samples:
            break
        if args.dataset in ['mug']:
            img, action, subject = data
        else:
            img, action, subject = data[0], None, None
        a = process_latent.sampling(sampling_number=img.shape[0])
        if args.first_stage_model:
            b, v, c, h, w = img.shape
            data = vq_model.encode(img.view(-1, c, args.input_size, args.input_size).to(device=device)).cpu()
            data = data.view(b, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
        else:
            data = img
        action, subject, x_fix_s, x_fix_d = swap_with_ldm_sample(args, data, a, action, subject,
                                                                 device, model,
                                                                 shape)
        if args.first_stage_model:
            x_fix_s = vq_model.decode(x_fix_s.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).view(
                img.shape).cpu()
            x_fix_d = vq_model.decode(
                x_fix_d.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).view(img.shape).cpu()
        for j in range(img.shape[0]):
            example_fix_s = torch.cat([img[[j]], x_fix_s[[j]].cpu()], dim=0)
            example_fix_d = torch.cat([img[[j]], x_fix_d[[j]].cpu()], dim=0)
            example_fix_s = torch.clip(example_fix_s, min=-1, max=1)
            example_fix_d = torch.clip(example_fix_d, min=-1, max=1)
            grid_fix_s = (make_grid(example_fix_s.view(-1, *img.shape[-3:]), nrow=args.video_length) + 1) / 2
            grid_fix_d = (make_grid(example_fix_d.view(-1, *img.shape[-3:]), nrow=args.video_length) + 1) / 2
            logger.log_fig(f'sample/sample_s_by_ldm', grid_fix_s.permute(1, 2, 0).numpy())
            logger.log_fig(f'sample/sample_d_by_ldm', grid_fix_d.permute(1, 2, 0).numpy())


def eval_fix(args, device, logger, model, current_epoch, vq_model=None):
    if current_epoch < args.save_epochs:
        return
    copy_args = copy.deepcopy(args)
    number_of_saves = current_epoch // copy_args.save_epochs
    epochs = number_of_saves * copy_args.save_epochs
    args.epoch = epochs
    args.mode = 'save_latent'
    save_latent(args, device, model)
    args.save_epochs = args.epochs_latent
    args.is_latent = True
    args.deterministic = True
    args.mode = 'train_latent_ddim'
    model2 = train_latent_ddim(args, logger)
    args.mode = 'eval_classification_on_ldm'
    shape = get_dataset_config(args)
    sample_fix(args, logger, model, model2, shape, device, vq_model=vq_model)
    if args.dataset in ['mug']:
        to_log, total_steps = eval_classification_on_ldm(args, model, model2, device, shape)
        for key, value in to_log.items():
            logger.log_name_params(f'eval_fix_ldm/{key}', value / total_steps)


def tsne_test(args, device, evalloader, logger, model):
    if args.dataset in ['mug']:
        ss, ds = [], []
        actions = []
        subjects = []
        with torch.no_grad():
            for data in tqdm(evalloader, desc="TSNE"):
                img, action, subject = data
                if args.model in ['timediffpriorkarras']:
                    s, d, _ = model.encoder(img.to(device))
                ss.append(s)
                ds.append(d)
                actions.append(action)
                subjects.append(subject)
            ss = torch.cat(ss, dim=0).cpu().numpy()
            subjects = torch.cat(subjects).detach().cpu().numpy()
            ds = torch.cat(ds, dim=0)
            d_mean = ds.mean(dim=1).cpu().numpy()
            v = ds.shape[1]
            ds = ds.view(-1, d.shape[-1]).cpu().numpy()
            actions = torch.cat(actions)
            actions_repeat = actions.repeat_interleave(v).cpu().numpy()
            actions = actions.cpu().numpy()
            plot_tsen_static(logger, ss, subjects)
            plot_tsne_dynamic(actions, actions_repeat, d_mean, ds, logger)

@torch.no_grad()
def save_latent(args, device, model, vq_model=None):
    if args.dataset in ['timit', 'libri']:
        if args.mel:
            resampler = torchaudio.transforms.Resample(orig_freq=16000 if args.dataset == 'timit' else 16000, new_freq=22050)
            spectrogram = mel_spectrogram
        else:
            spectrogram = torchaudio.transforms.Spectrogram(n_fft=args.fft_size, win_length=args.w_len, hop_length=args.h_len,
                                                        power=args.power).to(device)
            resampler = None

    else:
        resampler = None
        spectrogram = None
    all_a, all_s, all_d = [], [], []
    all_action, all_subject = [], []
    if args.first_stage_model and args.latent_dataset:
        dataloader = get_latent(args)
    else:
        if args.dataset in ['etth', 'physionet', 'airq']:
            dataloader, _, _ = get_dataset(args)
        else:
            dataloader, _ = get_dataset(args)
    for idx, data in tqdm(enumerate(dataloader), desc="Save Latent", total=len(dataloader)):
        if args.dataset in ['mug']:
            img, action, subject = data
        elif args.dataset in ['timit', 'libri']:
            if args.mel:
                data, ch_num = audio_path_to_data_hifi(data['wav'], resampler)
            else:
                data, _ = audio_path_to_data(data['wav'])
            data = data.to(device=device)
            data = spectrogram(data)
            data = data.permute(0, 2, 1)
            if args.dataset == 'timit':
                data = timit_normalize(args, data)
            elif args.dataset == 'libri':
                data = libri_normalize(args, data)
            img = data.cpu()
            subject, action = None, None
        else:
            img = data[0]
            subject, action = None, None
        if action is not None:
            all_action.append(action)
        if subject is not None:
            all_subject.append(subject)
        img = img.to(device=device)
        if args.first_stage_model and not args.latent_dataset and args.dataset not in ['timit', 'libri'] :
            b, v, c, h, w = img.shape
            data = vq_model.encode(img.view(-1, *img.shape[2:]).to(device=device)).cpu()
            data = data.view(b, v, *data.shape[1:])
        else:
            data = img
        if args.model in ['timediffpriorkarras']:
            s, d, a = model.encoder(data)
        else:
            raise NotImplementedError("Model not supported")
        all_a.append(a.detach().cpu().numpy())
        all_s.append(s.detach().cpu().numpy())
        all_d.append(d.detach().cpu().numpy())
    all_a = np.concatenate(all_a)
    all_s = np.concatenate(all_s)
    all_d = np.concatenate(all_d)
    to_save = {'all_a': all_a, 'all_s': all_s, 'all_d': all_d}
    if len(all_action) > 0:
        all_action = np.concatenate(all_action)
        to_save['all_action'] = all_action
    if len(all_subject) > 0:
        all_subject = np.concatenate(all_subject)
        to_save['all_subject'] = all_subject
    np.savez("{}_{}_latent".format(args.model, generate_exp_string(args).replace(".", "_")),
             **to_save)


def train_latent_ddim(args, logger: BaseLogger = None, num_layers=10, normalize=False):
    latent_dir = getattr(args, 'latent_dir', None) or os.getcwd()
    latent_file = os.path.join(latent_dir, '{}_{}_latent.npz'.format(
        args.model, generate_exp_string(args).replace('.', '_')))
    if args.dataset in ['mug', 'vox1', 'celebv', 'taichi', 'etth', 'airq', 'physionet', 'timit', 'libri']:
        dataset = LatentTimeDataset(latent_file, normalize=normalize)
    else:
        dataset = LatentDataset(latent_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    seed_everything(args.r_seed)
    if logger is None:
        logger = TqdmLogger()
        if args.tags is not None:
            logger.add_tags([*args.tags, 'latent_train'])
        logger.log_hparams(dict(vars(args)))
    old_shape = get_dataset_config(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset in ['timit', 'libri']:
        shape = (1, args.s_dim + args.d_dim * 68)
    elif args.dataset in ['mug', 'vox1', 'celebv', 'taichi', 'etth', 'physionet', 'airq']:
        shape = (1, args.s_dim + args.d_dim * old_shape[0], args.s_dim + args.d_dim * old_shape[0])
    else:
        shape = (1, args.a_dim, args.a_dim)
    if args.latent_const:
        model = LatentDiffConst(args, device, shape, num_layers)
    elif args.latent_s_d_split:
        model = LatentDiffSplit(args, device, shape, num_layers)
    else:
        model = LatentDiff(args, device, shape, num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(args.epochs, [losses], prefix='Epoch ')
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2., warm_epoch=1, after_scheduler=cosineScheduler)
    global_step = 0
    epochs = args.epochs if args.epochs_latent <= 0 else args.epochs_latent
    for curr_epoch in trange(0, epochs, desc="Epoch #"):
        total_loss = 0
        batch_bar = tqdm(dataloader, desc="Batch #")
        for idx, data in enumerate(batch_bar):
            data = data.to(device=device)
            loss = model.loss_fn(args=args, x=data, curr_epoch=curr_epoch)
            batch_bar.set_postfix(loss=format(loss, '.4f'))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
            logger.log('train_ldm/loss', loss.item(), global_step)
        losses.update(total_loss / idx)
        current_epoch = curr_epoch
        progress.display(current_epoch)
        current_epoch += 1
        warmUpScheduler.step()
        losses.reset()
        if current_epoch % args.save_epochs == 0:
            save_model(args, current_epoch, model)
    model.eval()
    return model


def eval_classification_on_ldm(args, model, model2, device, shape):
    assert args.dataset in ['mug']
    _, evalloder = get_dataset(args)
    classifiar = get_classifier(args).to(device)
    to_log = {}
    total_steps = 0
    len_value = args.batch_size
    if args.model in ['timediffpriorkarras']:
        process_latent = LatentDiffusionProcess(args, model2, device)
        for i in tqdm(range(10), desc="Classification on LDM"):
            for data in tqdm(evalloder, desc="Batch #", total=len(evalloder)):
                to_sum = judge_sample(args, classifiar, model, data, process_latent, device, shape)
                for key in to_sum:
                    if key in to_log:
                        to_log[key] += to_sum[key].sum().item()
                    else:
                        to_log[key] = to_sum[key].sum().item()
                    len_value = len(to_sum[key])
                total_steps += len_value
            tqdm.write(f"Loop {i + 1} current results")
            for key, value in to_log.items():
                value = value / total_steps
                tqdm.write(f"{key}: {100 * value:.4f}%")
        print("Final Results")
        for key, value in to_log.items():
            value = value / total_steps
            print(f"{key}: {100 * value:.4f}%")
    return to_log, total_steps


def swap_with_ldm_sample(args, img, sample_a, action, subject, device, model, shape, process=None, use_xt=False):
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    b, v, *_ = img.shape
    img = img.to(device=device)
    sample_s, sample_d = sample_a[:, :args.s_dim], sample_a[:, args.s_dim:]
    sample_s = sample_s.unsqueeze(1).expand(b, v, args.s_dim)
    sample_d = sample_d.view(b, v, args.d_dim)
    s, d, a_og = model.encoder(img)
    s_expand = s.unsqueeze(1).expand(b, v, args.s_dim)
    a_sample_fix_s = torch.cat((sample_d, s_expand), dim=2)
    a_sample_fix_d = torch.cat((d, sample_s), dim=2)
    a = torch.cat((a_sample_fix_s, a_sample_fix_d), dim=0)
    a = a.reshape(2 * b * v, -1)
    if use_xt:
        x_t = process.reverse_sampling(torch.cat((img, img), 0), torch.cat((a_og, a_og), 0))
    else:
        x_t = None
    x_rec = process.sampling(sampling_number=b * 2, xT=x_t, a=a)
    x_fix_s, x_fix_d = x_rec[:b], x_rec[b:]
    return action, subject, x_fix_s, x_fix_d

@torch.no_grad()
def swap_with_ldm_sample_vq(args, img, sample_a, action, subject, device, model, vq_model, shape, process=None, use_xt=True):
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    b, v, *_ = img.shape
    img = img.to(device=device)
    if args.first_stage_model:
        b, v, c, h, w = img.shape
        data = vq_model.encode(img.view(-1, *img.shape[2:]).to(device=device)).cpu()
        data = data.view(b, v, *data.shape[1:]).to(device)
    else:
        data = img
    sample_s, sample_d = sample_a[:, :args.s_dim], sample_a[:, args.s_dim:]
    sample_s = sample_s.unsqueeze(1).expand(b, v, args.s_dim)
    sample_d = sample_d.view(b, v, args.d_dim)
    s, d, a_og = model.encoder(data)
    s_expand = s.unsqueeze(1).expand(b, v, args.s_dim)
    a_sample_fix_s = torch.cat((sample_d, s_expand), dim=2)
    a_sample_fix_d = torch.cat((d, sample_s), dim=2)
    a = torch.cat((a_sample_fix_s, a_sample_fix_d), dim=0)
    a = a.reshape(2 * b * v, -1)
    if use_xt:
        x_t = process.reverse_sampling(torch.cat((data, data), 0), torch.cat((a_og, a_og), 0))
    else:
        x_t = None
    x_rec = process.sampling(sampling_number=b * 2, xT=x_t, a=a)
    if args.first_stage_model:
        x_rec = vq_model.decode(
            x_rec.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).view(b * 2 * v, c, h, w).cpu()
    x_rec = torch.clip(x_rec.view(b*2, v, c, h, w), -1, 1).cpu()
    x_fix_s, x_fix_d = x_rec[:b], x_rec[b:]
    return action, subject, x_fix_s, x_fix_d


def judge_sample_mug(args, classifiar, model, data, process_latent, device, shape):
    img, action, subject = data
    a = process_latent.sampling(sampling_number=img.shape[0])
    action, subject, x_fix_s, x_fix_d = swap_with_ldm_sample(args, img, a, action, subject,
                                                             device, model,
                                                             shape)
    fix_s_action, fix_s_subject = classifiar(denormalize(x_fix_s))
    fix_d_action, fix_d_subject = classifiar(denormalize(x_fix_d))
    fix_s_action = torch.softmax(fix_s_action, dim=1).argmax(-1).cpu()
    fix_s_subject = torch.softmax(fix_s_subject, dim=1).argmax(-1).cpu()
    fix_d_action = torch.softmax(fix_d_action, dim=1).argmax(-1).cpu()
    fix_d_subject = torch.softmax(fix_d_subject, dim=1).argmax(-1).cpu()
    subject_fix_s = fix_s_subject.cpu() == subject
    subject_fix_d = fix_d_subject.cpu() == subject
    action_fix_d = fix_d_action == action
    action_fix_s = fix_s_action == action
    return {'subject_acc_fix_s': subject_fix_s, 'subject_acc_fix_d': subject_fix_d,
            'action_acc_fix_d': action_fix_d, 'action_acc_fix_s': action_fix_s}


@torch.no_grad()
def judge_sample(args, classifiar, model: DiffSDAPriorKarras
                 , data, process_latent, device, shape=None):
    if args.dataset in ['mug'] and args.model in ['timediffpriorkarras']:
        return judge_sample_mug(args, classifiar, model, data, process_latent, device, shape)


def plot_tsne_dynamic(actions, actions_repeat, d_mean, ds, logger):
    d_tsne = manifold.TSNE(n_components=2).fit_transform(ds)
    d_mean_tsne = manifold.TSNE(n_components=2).fit_transform(d_mean)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    id_to_cls = {
        0: "anger",
        1: "disgust",
        2: "happiness",
        3: "fear",
        4: "sadness",
        5: "surprise",
    }
    colors = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598',
              '#99d594', '#3288bd']
    for i in range(6):
        mask = np.array(actions_repeat) == i
        mask_mean = np.array(actions) == i
        ax[0].scatter(d_tsne[mask, 0], d_tsne[mask, 1], c=colors[i], label=id_to_cls[i],
                      alpha=.75)
        ax[1].scatter(d_mean_tsne[mask_mean, 0], d_mean_tsne[mask_mean, 1], c=colors[i], label=id_to_cls[i],
                      alpha=.75)
    ax[0].locator_params(nbins=4)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_rasterized(True)
    ax[1].set_rasterized(True)
    title1 = 'dynamic variable'
    title2 = 'dynamic variable mean'
    ax[0].set_title(title1)
    ax[1].set_title(title2)
    logger.log_fig('latent/d_tsne', fig)


def plot_tsen_static(logger, ss, subjects):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    s_tsne = manifold.TSNE(n_components=2).fit_transform(ss)
    N = np.max(subjects) + 1
    colors = np.array(distinctipy.get_colors(N))
    ax.scatter(s_tsne[:, 0], s_tsne[:, 1], c=colors[subjects], alpha=.75)
    logger.log_fig('latent/s_tsne', fig)


def swap_eval(args, classifiar, current_epoch, device, evalloader, logger, model, shape, num_samples=100):
    to_log = {}
    data_iter = iter(evalloader)
    if num_samples == 0:
        return False
    if num_samples > len(evalloader.dataset) or num_samples < 0:
        num_samples = len(evalloader.dataset)
    num_iter = num_samples // args.batch_size
    with torch.no_grad():
        if args.dataset in ['mug']:
            total_samples = 0
            for _ in trange(0, num_iter, desc="Swap Eval"):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(evalloader)
                    data = next(data_iter)
                to_sum = judge_swap(args, classifiar, model, data, device, shape)
                for key, value in to_sum.items():
                    if key not in to_log:
                        to_log[key] = 0
                    to_log[key] += value.sum().item()
                    value_len = len(value)
                total_samples += value_len
        else:
            raise NotImplementedError
    for key, value in to_log.items():
        logger.log(f'eval/{key}', value / total_samples, current_epoch)
    # return condition_to_stop(args, current_epoch, {key: value / total_samples for key, value in to_log.items()})
    return False

def a_sampling(data, model):
    _, _, a = model.encoder(data)
    return a

def judge_swap(args, classifiar, model: DiffSDAPriorKarras
               , data, device, shape=None):
    if args.dataset in ['mug'] and args.model in ['timediffpriorkarras']:
        return judge_swap_mug(args, classifiar, model, data, device, shape)


@torch.no_grad()
def judge_swap_mug(args, classifiar: classifier_MUG, model: DiffSDAPriorKarras
                   , data, device, shape=None):
    img, action, subject = data
    action, action_roll, subject, subject_roll, x_rec, x_rec_og = swap_sample(args, img, action, subject, device, model,
                                                                              shape)
    rec_action, rec_subject = classifiar(denormalize(x_rec))
    rec_action_og, rec_subject_og = classifiar(denormalize(x_rec_og))
    rec_action = torch.softmax(rec_action, dim=1).argmax(-1).cpu()
    rec_subject = torch.softmax(rec_subject, dim=1).argmax(-1).cpu()
    rec_action_og = torch.softmax(rec_action_og, dim=1).argmax(-1).cpu()
    rec_subject_og = torch.softmax(rec_subject_og, dim=1).argmax(-1).cpu()
    subject_fix_s = rec_subject.cpu() == subject
    subject_fix_d = rec_subject.cpu() == subject_roll
    subject_og = rec_subject_og.cpu() == subject
    action_fix_d = rec_action == action_roll
    action_fix_s = rec_action == action
    action_og = rec_action_og == action
    return {'subject_acc_fix_s': subject_fix_s, 'subject_acc_fix_d': subject_fix_d, 'subject_acc_og': subject_og,
            'action_acc_fix_d': action_fix_d, 'action_acc_fix_s': action_fix_s, 'action_acc_og': action_og}


@torch.no_grad()
def log_swap_sample(args, data, model, device, shape=None, logger: BaseLogger = None, vq_model=None, extra_text="", num_example=4, process=None, use_xt=False):
    if args.dataset in ['mug']:
        img, action, subject = data
        img, action, subject = img[:num_example], action[:num_example], subject[:num_example]
    elif args.dataset in ['taichi', 'vox1', 'celebv']:
        img, action, subject = data[0][:num_example], None, None
        img = img[:args.sampling_number]
    else:
        return
    if args.first_stage_model:
        b, v, c, h, w = img.shape
        data = vq_model.encode(img.view(-1, c, args.input_size, args.input_size).to(device=device)).cpu()
        data = data.view(b, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size)
    else:
        data = img
    _, _, _, _, x_rec, _ = swap_sample(args, data, action, subject, device, model, shape, process=process, use_xT=use_xt)
    _, _, _, _, x_rec_rev, _ = swap_sample(args, data, action, subject, device, model, shape, rev_roll=True, process=process, use_xT=use_xt)
    if args.first_stage_model:
        x_rec = vq_model.decode(x_rec.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).view(img.shape).cpu()
        x_rec_rev = vq_model.decode(x_rec_rev.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).view(img.shape).cpu()
    else:
        x_rec = x_rec.cpu()
        x_rec_rev = x_rec_rev.cpu()
    for i in range(0, img.shape[0]-1):
        example = torch.cat([img[i:i + 2], x_rec[[i + 1]], x_rec_rev[[i]]], dim=0)
        example = torch.clip(example, min=-1, max=1)
        grid = (make_grid(example.view(-1, *img.shape[-3:]), nrow=args.video_length) + 1) / 2
        logger.log_fig(f'sample/swap{i}{extra_text}', grid.permute(1, 2, 0).numpy())


def swap_sample(args, img, action, subject, device, model, shape, rev_roll=False, process=None, use_xT=True):
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    b, v, *_ = img.shape
    roll_by = -1 if rev_roll else 1
    action_roll = torch.roll(action, roll_by, dims=0) if action is not None else None
    subject_roll = torch.roll(subject, roll_by, dims=0) if subject is not None else None
    img = img.to(device=device)
    if args.model in ['timediffpriorkarras']:
        s, d, a = model.encoder(img)
    img.cpu()
    og_a = a
    if args.sheared_s:
        s_expand = s.unsqueeze(1).expand(b, v, args.s_dim)
    else:
        s_expand = s
    d_roll = torch.roll(d, roll_by, dims=0)
    a = torch.cat((d_roll, s_expand), dim=2)
    a2 = torch.cat((d, s_expand), dim=2)
    a = a.reshape(b * v, -1)
    a2 = a2.reshape(b * v, -1)
    a_with_og = torch.cat((a, og_a), dim=0)
    if use_xT:
        x_t = process.reverse_sampling(torch.cat((img, img), 0).to(device), torch.cat((a2, og_a), dim=0))
        x_t_rec, x_t_rec_og = x_t[:b], x_t[b:]
        x_t_rec = torch.roll(x_t_rec, roll_by, dims=0)
        x_t = torch.cat((x_t_rec, x_t_rec_og), 0)
    else:
        x_t = None
    x_rec = process.sampling(sampling_number=b * 2, xT=x_t, a=a_with_og)
    x_rec, x_rec_og = x_rec[:b], x_rec[b:]
    return action, action_roll, subject, subject_roll, x_rec, x_rec_og


@torch.no_grad()
def sample_cond_images(args, model, data, device, shape=None, logger: BaseLogger = None, epoch=0, vq_model=None, process=None):
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    data = data[:args.sampling_number]
    b, v, c, h, w = data.shape
    img = data
    data = data.to(device=device)
    if args.first_stage_model:
        data = vq_model.encode(data.view(-1, c, h, w))
        data = data.view(b, v, args.z_channels_vq, args.vq_input_size, args.vq_input_size).detach()
    a = a_sampling(data, model)
    if args.batch_size == 1:
        xT_original = torch.randn_like(data).repeat(args.sampling_number, *(1 for _ in range(len(data.shape) - 1)))
        a_original = a.repeat(args.sampling_number, 1)
    else:
        xT_original = process.reverse_sampling(data, a)
        a_original = a
    batch = process.sampling(args.sampling_number, xT=xT_original, a=a_original)
    if args.first_stage_model:
        batch = vq_model.decode(batch.view(-1, args.z_channels_vq, args.vq_input_size, args.vq_input_size)).detach().cpu().view(b, v, c, h, w)
    if logger is not None:
        log_sample(args, img, logger, name='sample/real')
    imgs = torch.clip(batch, min=-1, max=1).cpu()
    if logger is not None and len(imgs) > 0:
        if args.dataset in ['mug']:
            imgs = imgs.view(-1, *imgs[0].shape[-3:])
        log_sample(args, imgs, logger, name='sample/cond')


def log_sample(args, data, logger, name):
    grid = (make_grid(data.detach().cpu().view(-1, *data.shape[-3:]), nrow=args.video_length) + 1) / 2
    logger.log_fig(name, grid.permute(1, 2, 0).numpy())


@torch.no_grad()
def sample_cond_timit(args, model, data, device, shape=None, logger: BaseLogger = None, process=None, use_xt=False):
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    data = data[:args.sampling_number]
    b, t, f = data.shape
    img = data
    data = data.to(device=device)
    a = a_sampling(data, model)
    if args.batch_size == 1:
        if use_xt:
            xT_original = process.reverse_sampling(data, a).repeat(args.sampling_number, *(1 for _ in range(len(data.shape) - 1)))
        else:
            xT_original = torch.randn_like(data).repeat(args.sampling_number, *(1 for _ in range(len(data.shape) - 1)))
        a_original = a.repeat(args.sampling_number, 1)

    else:
        if use_xt:
            xT_original = process.reverse_sampling(data, a)
        else:
            xT_original = torch.randn_like(data)
        a_original = a
    batch = process.sampling(xT=xT_original, a=a_original)
    if logger is not None:
        if args.dataset in ['timit']:
            img = img.view(b, t, f)
            grid = (make_grid(img.detach().cpu().view(-1, *img.shape[-2:])[:, None], nrow=args.video_length) + 1) / 2
            logger.log_fig('sample/real', grid[[0]].permute(1, 2, 0).numpy()[..., 0])
        else:
            log_sample(args, img, logger, name='sample/real')
    imgs = batch.cpu()
    if logger is not None and len(imgs) > 0:
        if args.dataset in ['mug']:
            imgs = imgs.view(-1, *imgs[0].shape[-3:])
        if args.dataset in ['timit']:
            imgs = imgs.view(b, t, f)
            grid = (make_grid(imgs.detach().cpu().view(-1, *imgs.shape[-2:])[:, None], nrow=args.video_length) + 1) / 2
            logger.log_fig('sample/cond', grid[[0]].permute(1, 2, 0).numpy()[..., 0])
        else:
            log_sample(args, imgs, logger, name='sample/cond')


@torch.no_grad()
def save_timit_audio(args, model, data, ch_num, device, shape, logger=None, process=None, use_xt=False, hifigan=None,
                     denoiser=None):
    ispectogram = torchaudio.transforms.GriffinLim(n_fft=args.fft_size, win_length=args.w_len, hop_length=args.h_len,
                                                   length=3200, power=1).to(device)
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    ch_num = ch_num[:args.sampling_number]
    examples = np.sum(ch_num)
    data = data[:examples]
    b, t, f = data.shape
    a = a_sampling(data, model)
    if args.batch_size == 1:
        if use_xt:
            xT_original = process.reverse_sampling(data, a).repeat(args.sampling_number, *(1 for _ in range(len(data.shape) - 1)))
        else:
            xT_original = torch.randn_like(data).repeat(args.sampling_number, *(1 for _ in range(len(data.shape) - 1)))
        a_original = a.repeat(args.sampling_number, 1)
    else:
        if use_xt:
            xT_original = process.reverse_sampling(data, a)
        else:
            xT_original = torch.randn_like(data)
        a_original = a
    batch = process.sampling(xT=xT_original, a=a_original)
    if args.dataset in ['timit']:
        data = timit_denormalize(args, data)
        batch = timit_denormalize(args, batch)
    elif args.dataset in ['libri']:
        data = libri_denormalize(args, data)
        batch = libri_denormalize(args, batch)
    os.makedirs('test_audio', exist_ok=True)
    ch_sum = np.cumsum(ch_num)
    ch_sum = np.concatenate([np.array([0]), ch_sum])
    if args.mel:
        input_waveform = hifigan(data.permute(0, 2, 1)).float()
        input_waveform = denoiser(input_waveform.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_waveform = hifigan(batch.permute(0, 2, 1)).float()
        output_waveform = denoiser(output_waveform.squeeze(1), 0.005).squeeze(1).cpu().numpy()
    else:
        input_waveform = ispectogram(data.permute(0, 2, 1)).cpu()
        output_waveform = ispectogram(batch.permute(0, 2, 1)).cpu()
    for i, (s, e) in enumerate(zip(ch_sum[:-1], ch_sum[1:])):
        if args.mel:
            write(f"test_audio/input_{i}.wav", 22050, input_waveform[s:e].reshape(-1))
            write(f"test_audio/output_{i}{'_XT' if use_xt else ''}.wav", 22050, output_waveform[s:e].reshape(-1))
        else:
            torchaudio.save(f"test_audio/input_{i}.wav", input_waveform[s:e].reshape(1, -1), 16000)
            torchaudio.save(f"test_audio/output_{i}{'_XT' if use_xt else ''}.wav", output_waveform[s:e].reshape(1, -1), 16000)
        if logger is not None:
            logger.log_audio(f"audio/input_{i}", f"test_audio/input_{i}.wav")
            logger.log_audio(f"audio/output_{i}{'_XT' if use_xt else ''}",
                             f"test_audio/output_{i}{'_XT' if use_xt else ''}.wav")


@torch.no_grad()
def save_timit_audio_swap(args, model, data, ch_num, device, shape, logger=None, process=None, use_xt=False,
                          hifigan=None, denoiser=None):
    ispectogram = torchaudio.transforms.GriffinLim(n_fft=args.fft_size, win_length=args.w_len, hop_length=args.h_len,
                                                   length=3200, power=1).to(device)
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    ch_num = ch_num[:2]
    examples = np.sum(ch_num)
    data = data[:examples]
    example0 = data[:ch_num[0]].reshape(1, -1, data.shape[-1])
    example1 = data[ch_num[0]:ch_num[1]+ch_num[0]].reshape(1, -1, data.shape[-1])
    s0, d0, a0 = model.encoder(example0)
    s1, d1, a1 = model.encoder(example1)
    og_a0 = a0
    og_a1 = a1
    if args.no_time:
        raise NotImplementedError("Not supported")
    else:
        s0 = s0[:, None].expand(1, d1.shape[1], args.s_dim)
        s1 = s1[:, None].expand(1, d0.shape[1], args.s_dim)
        a0 = torch.cat((d0, s1), dim=2)
        a1 = torch.cat((d1, s0), dim=2)
    if use_xt:
        x_t0 = process.reverse_sampling(example0, og_a0)
        x_t1 = process.reverse_sampling(example1, og_a1)
    else:
        x_t0 = torch.randn(example0.shape).to(device)
        x_t1 = torch.randn(example1.shape).to(device)

    batch0 = process.sampling(sampling_number=1, xT=x_t0, a=a0)
    batch1 = process.sampling(sampling_number=1, xT=x_t1, a=a1)
    denorm = timit_denormalize if args.dataset in ['timit'] else libri_denormalize
    example0 = denorm(args, example0)
    example1 = denorm(args, example1)
    batch0 = denorm(args, batch0)
    batch1 = denorm(args, batch1)
    os.makedirs('test_audio', exist_ok=True)
    if args.mel:
        input_waveform0 = hifigan(example0.permute(0, 2, 1)).float()
        input_waveform1 = hifigan(example1.permute(0, 2, 1)).float()
        input_waveform0 = denoiser(input_waveform0.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        input_waveform1 = denoiser(input_waveform1.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_waveform0 = hifigan(batch0.permute(0, 2, 1)).float()
        output_waveform0 = denoiser(output_waveform0.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_waveform1 = hifigan(batch1.permute(0, 2, 1)).float()
        output_waveform1 = denoiser(output_waveform1.squeeze(1), 0.005).squeeze(1).cpu().numpy()
    else:
        input_waveform0 = ispectogram(example0.permute(0, 2, 1)).cpu()
        input_waveform1 = ispectogram(example1.permute(0, 2, 1)).cpu()
        output_waveform0 = ispectogram(batch0.permute(0, 2, 1)).cpu()
        output_waveform1 = ispectogram(batch1.permute(0, 2, 1)).cpu()
    if args.mel:
        write(f"test_audio/input_for_swap_{0}.wav", 22050, input_waveform0.reshape(-1))
        write(f"test_audio/input_for_swap_{1}.wav", 22050, input_waveform1.reshape(-1))
        write(f"test_audio/output_for_swap_{0}{'_XT' if use_xt else ''}.wav", 22050, output_waveform0.reshape(-1))
        write(f"test_audio/output_for_swap_{1}{'_XT' if use_xt else ''}.wav", 22050, output_waveform1.reshape(-1))
    else:
        torchaudio.save(f"test_audio/input_for_swap_{0}.wav", input_waveform0.reshape(1, -1), 16000)
        torchaudio.save(f"test_audio/input_for_swap_{1}.wav", input_waveform1.reshape(1, -1), 16000)
        torchaudio.save(f"test_audio/output_for_swap_{0}{'_XT' if use_xt else ''}.wav", output_waveform0.reshape(1, -1), 16000)
        torchaudio.save(f"test_audio/output_for_swap_{1}{'_XT' if use_xt else ''}.wav", output_waveform1.reshape(1, -1), 16000)
    if logger is not None:
        for i in range(2):
            logger.log_audio(f"audio/input_for_swap_{i}", f"test_audio/input_for_swap_{i}.wav")
            logger.log_audio(f"audio/output_for_swap_{i}{'_XT' if use_xt else ''}",
                             f"test_audio/output_for_swap_{i}{'_XT' if use_xt else ''}.wav")
@torch.no_grad()
def save_for_rebutle_audio_swap(args, model, data, id_A, id_B, index, ch_num, device, shape, process=None, use_xt=True,
                          hifigan=None, denoiser=None):
    base_path = 'timit_demo'
    ispectogram = torchaudio.transforms.GriffinLim(n_fft=args.fft_size, win_length=args.w_len, hop_length=args.h_len,
                                                   length=3200, power=1).to(device)
    shape = get_dataset_config(args) if shape is None else shape
    process = DiffusionProcess(args, model, device, shape) if process is None else process
    ch_num = ch_num[:2]
    examples = np.sum(ch_num)
    data = data[:examples]
    example0 = data[:ch_num[0]].reshape(1, -1, data.shape[-1])
    example1 = data[ch_num[0]:ch_num[1]+ch_num[0]].reshape(1, -1, data.shape[-1])
    s0, d0, a0 = model.encoder(example0)
    s1, d1, a1 = model.encoder(example1)
    og_a0 = a0
    og_a1 = a1
    if args.no_time:
        raise NotImplementedError("Not supported")
    else:
        s0 = s0[:, None].expand(1, d1.shape[1], args.s_dim)
        s1 = s1[:, None].expand(1, d0.shape[1], args.s_dim)
        a0 = torch.cat((d0, s1), dim=2)
        a1 = torch.cat((d1, s0), dim=2)
    if use_xt:
        x_t0 = process.reverse_sampling(example0, og_a0)
        x_t1 = process.reverse_sampling(example1, og_a1)
    else:
        x_t0 = torch.randn(example0.shape).to(device)
        x_t1 = torch.randn(example1.shape).to(device)
    batch0 = process.sampling(sampling_number=1, xT=x_t0, a=a0)
    batch1 = process.sampling(sampling_number=1, xT=x_t1, a=a1)
    recon0 = process.sampling(sampling_number=1, xT=x_t0, a=og_a0)
    recon1 = process.sampling(sampling_number=1, xT=x_t1, a=og_a1)
    denorm = timit_denormalize if args.dataset in ['timit'] else libri_denormalize
    example0 = denorm(args, example0)
    example1 = denorm(args, example1)
    batch0 = denorm(args, batch0)
    batch1 = denorm(args, batch1)
    recon0 = denorm(args, recon0)
    recon1 = denorm(args, recon1)
    os.makedirs(base_path, exist_ok=True)
    if args.mel:
        input_waveform0 = hifigan(example0.permute(0, 2, 1)).float()
        input_waveform1 = hifigan(example1.permute(0, 2, 1)).float()
        input_waveform0 = denoiser(input_waveform0.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        input_waveform1 = denoiser(input_waveform1.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_waveform0 = hifigan(batch0.permute(0, 2, 1)).float()
        output_waveform1 = hifigan(batch1.permute(0, 2, 1)).float()
        output_waveform0 = denoiser(output_waveform0.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_waveform1 = denoiser(output_waveform1.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_recon0 = hifigan(recon0.permute(0, 2, 1)).float()
        output_recon1 = hifigan(recon1.permute(0, 2, 1)).float()
        output_recon0 = denoiser(output_recon0.squeeze(1), 0.005).squeeze(1).cpu().numpy()
        output_recon1 = denoiser(output_recon1.squeeze(1), 0.005).squeeze(1).cpu().numpy()
    else:
        raise NotImplementedError("Not supported")
    if args.mel:
        write(f"{base_path}/{index}_orig_{id_A}.wav", 22050, input_waveform0.reshape(-1))
        write(f"{base_path}/{index}_orig_{id_B}.wav", 22050, input_waveform1.reshape(-1))
        write(f"{base_path}/{index}_recon_{id_A}_our.wav", 22050, output_recon0.reshape(-1))
        write(f"{base_path}/{index}_recon_{id_B}_our.wav", 22050, output_recon1.reshape(-1))
        write(f"{base_path}/{index}_{id_A}_{id_B}_our.wav", 22050, output_waveform0.reshape(-1))
        write(f"{base_path}/{index}_{id_B}_{id_A}_our.wav", 22050, output_waveform1.reshape(-1))
    else:
        raise NotImplementedError("Not supported")
