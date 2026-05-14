# DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities

<div align="center">

**Hedi Zisling**, **Ilan Naiman**, **Nimrod Berman**, **Supasorn Suwajanakorn**, **Omri Azencot**

<a href="https://arxiv.org/abs/2510.05717"><img src="https://img.shields.io/badge/arXiv-2510.05717-b31b1b.svg" alt="arXiv"></a>
<a href="https://openreview.net/forum?id=tooDJHBSvO"><img src="https://img.shields.io/badge/OpenReview-DiffSDA-8c1b13.svg" alt="OpenReview"></a>

</div>

---

## 📢 News
*   **[Feb 2026]** 🎉 Accepted to **ICLR 2026**!
*   **[Oct 2025]** Paper released on arXiv.

---

## 🖼️ Teaser

<p align="center">
  <img src="figs/teaser_lr_opt.jpg" width="100%">
</p>

## 📄 Abstract

Unsupervised representation learning, particularly sequential disentanglement, aims to separate static and dynamic factors of variation in data without relying on labels. This remains a challenging problem, as existing approaches based on variational autoencoders and generative adversarial networks often rely on multiple loss terms, complicating the optimization process. Furthermore, sequential disentanglement methods face challenges when applied to real-world data, and there is currently no established evaluation protocol for assessing their performance in such settings. Recently, diffusion models have emerged as state-of-the-art generative models, but no theoretical formalization exists for their application to sequential disentanglement. In this work, we introduce the Diffusion Sequential Disentanglement Autoencoder (DiffSDA), a novel, modal-agnostic framework effective across diverse real-world data modalities, including time series, video, and audio. DiffSDA leverages a new probabilistic modeling, latent diffusion, and efficient samplers, while incorporating a challenging evaluation protocol for rigorous testing. Our experiments on diverse real-world benchmarks demonstrate that DiffSDA outperforms recent state-of-the-art methods in sequential disentanglement.

## 🏗️ Method

<p align="center">
  <img src="figs/pipeline_train.jpg" width="100%">
</p>

---

## Table of Contents

1. [Installation](#installation)
2. [Environment Variables](#environment-variables)
3. [Pre-trained Weights](#pre-trained-weights)
4. [Dataset Setup](#dataset-setup)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Citation](#citation)

---

## Installation

Tested on Linux (RHEL 9) with Python 3.10 and CUDA 11.8.

### 1. Clone the repo with submodules

```bash
git clone --recurse-submodules <repo-url> DiffSDA
cd DiffSDA
# If you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

The submodules under `OpenFacePytorch/`, `pose_estimation/`, and `reid_baseline/` are
only required for video evaluation (face / pose / re-identification metrics).
TaiChi evaluation additionally needs `pose_model.pth` and `reid_model.pth` —
see [Video — Reconstruction AKD / AED](#video--reconstruction-akd--aed-paper-table-3)
for the exact submodule commits and download links.

### 2. Create the conda environment

```bash
conda create -n diffsda python=3.10 -y
conda activate diffsda
```

### 3. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

For a different CUDA version, see https://pytorch.org/get-started/locally/.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` lists everything needed for training and evaluation across
all three modalities (video, audio, time-series).

### 5. Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from models import DiffSDAPriorKarras; print('OK')"
```

---

## Environment Variables

All dataset and model paths default to subdirectories of the repository root, so
the code works **out-of-the-box from a fresh clone**:

| Variable                   | Default                          | What it points to                                |
| -------------------------- | -------------------------------- | ------------------------------------------------ |
| `DIFFSDA_ROOT`             | repo root                        | project root used to derive every other path     |
| `DIFFSDA_DATASETS_ROOT`    | `$DIFFSDA_ROOT/data`             | raw and preprocessed datasets                    |
| `DIFFSDA_MODELS_ROOT`      | `$DIFFSDA_ROOT/checkpoints/runs` | training run output                              |
| `DIFFSDA_SAMPLES_ROOT`     | `$DIFFSDA_ROOT/samples`          | generated samples (NPZ / images / audio)         |
| `DIFFSDA_PRETRAINED_ROOT`  | `$DIFFSDA_ROOT/checkpoints/vq_models` | pre-trained VQ-VAE weights                  |
| `DIFFSDA_FINAL_WEIGHTS`    | `$DIFFSDA_ROOT/checkpoints/DiffSDA`   | released DiffSDA model weights              |
| `DIFFSDA_CLASSIFIERS_ROOT` | `$DIFFSDA_ROOT/checkpoints/classifiers` | evaluation classifiers (e.g. MUG)         |
| `DIFFSDA_LIBRI_ROOT`       | `$DIFFSDA_DATASETS_ROOT/LibriSpeech`  | LibriSpeech corpus (often a separate volume) |
| `DIFFSDA_LOGS_ROOT`        | `$DIFFSDA_ROOT/logs`             | Slurm / training logs                            |

Override any of them via your shell profile or before invoking a script:

```bash
export DIFFSDA_DATASETS_ROOT=/path/to/datasets
export DIFFSDA_FINAL_WEIGHTS=/path/to/checkpoints/DiffSDA
```

See `paths.py` for the full list of constants.

---

## Pre-trained Weights & Auxiliary Files

We host trained DiffSDA checkpoints, the first-stage VQ-VAE / KL-VAE weights,
the MUG evaluation classifier, and small dataset-side auxiliary files (latent
normalization stats, the processed PhysioNet CSVs, the CelebV-HQ filtered-clip
lists, LibriSpeech mel mean/std) on a public Google Drive folder. **Raw
datasets are not hosted** — see [Dataset Setup](#dataset-setup) for download
links to each dataset.

### Download

The release folder is here:

> 🗂️ <https://drive.google.com/drive/folders/1sJapoZrnuu4FmmlWWTeZUL5U4bxOcGMm?usp=sharing>

Files are uploaded individually so you can download just the pieces you need
(e.g. only `vox1.pth` + `vq-f8-n256_model_ft.ckpt` if you only run video
evaluation on VoxCeleb1). The Drive folder mirrors the layout the code
expects, so save each file to the matching path under your repo root.

### Folder layout (mirror of repo paths)

```
<Drive folder>/                        # ~5.9 GB total
  checkpoints/
    DiffSDA/                           # — released DiffSDA model weights
      vox1.pth                  924 MB   video    — VoxCeleb1
      celebv.pth                924 MB   video    — CelebV-HQ
      mug.pth                   107 MB   video    — MUG
      mug_small.pth             127 MB   video    — MUG (small variant)
      taichi.pth                315 MB   video    — TAICHI
      timeseries/
        physionet.pth            33 MB   time-series — PhysioNet 2012
        airq.pth                 52 MB   time-series — Beijing Air Quality
        etth.pth                 55 MB   time-series — ETT-h
    vq_models/                         # — first-stage autoencoders
      vq-f8-n256_model.ckpt     862 MB   used by --first_stage_model vq8
      vq-f8-n256_model_ft.ckpt  862 MB   used by --first_stage_model vq8ft (faces)
      vqf4_model.ckpt           721 MB   used by --first_stage_model vq4
      kl-f8.ckpt               1045 MB   used by --first_stage_model kl8
    classifiers/                       # — evaluation-only classifiers
      mug_cls_new_contrastive.tar   19 MB   MUG action classifier
  data/                                 (~7 MB of dataset-side auxiliary files)
    VoxCeleb/
      mean_std_256_vq{4,8,8ft}.npz    latent normalization stats
    CelebV-HQ/
      mean_std_256_vq{4,8,8ft}.npz
      filtered_clips.pkl              used by celebv_to_latent.py
      filtered_clips_new.pkl          train split index
      filtered_clips_new_test.pkl     test split index
    TAICHI/taichi-png/
      mean_std_256_vq8.npz
    LibriSpeech/
      mean_std.pkl                    mel-spec mean/std
    physionet/
      Outcomes-a.txt                  required by physionet loader
      processed_df.csv                skips ~5 min preprocessing on first run
      processed_static_df.csv
```

> No audio (TIMIT / LibriSpeech) DiffSDA checkpoints are released — please
> train from scratch with `train_audio.py` (see [Audio](#audio) below).

### What you actually need

Pick only the rows that match your modality / dataset:

| If you want to…                        | Download from `checkpoints/`                     | Download from `data/`              |
| -------------------------------------- | ------------------------------------------------ | ---------------------------------- |
| Evaluate video on VoxCeleb1            | `DiffSDA/vox1.pth`, `vq_models/vq-f8-n256_model_ft.ckpt` | `VoxCeleb/mean_std_256_vq8ft.npz` |
| Evaluate video on CelebV-HQ            | `DiffSDA/celebv.pth`, `vq_models/vq-f8-n256_model_ft.ckpt` | `CelebV-HQ/mean_std_256_vq8ft.npz` + `filtered_clips*.pkl` |
| Evaluate video on TAICHI               | `DiffSDA/taichi.pth`, `vq_models/vq-f8-n256_model.ckpt`   | `TAICHI/taichi-png/mean_std_256_vq8.npz` |
| Evaluate video on MUG                  | `DiffSDA/mug.pth`, `classifiers/mug_cls_new_contrastive.tar` | (none)                            |
| Evaluate TS on PhysioNet               | `DiffSDA/timeseries/physionet.pth`               | `physionet/{Outcomes-a.txt,processed_df.csv,processed_static_df.csv}` |
| Evaluate TS on AirQ / ETT-h            | `DiffSDA/timeseries/{airq,etth}.pth`             | (none)                            |
| Train audio on LibriSpeech from scratch| (no audio weights released — train yourself)     | `LibriSpeech/mean_std.pkl`        |

Save each file to the matching path under your repo root, e.g.

```bash
# example — set up a VoxCeleb1 evaluation
mkdir -p checkpoints/DiffSDA checkpoints/vq_models data/VoxCeleb
mv ~/Downloads/vox1.pth                 checkpoints/DiffSDA/
mv ~/Downloads/vq-f8-n256_model_ft.ckpt checkpoints/vq_models/
mv ~/Downloads/mean_std_256_vq8ft.npz   data/VoxCeleb/
```

After that, raw datasets you download yourself (VoxCeleb1 face crops, MUG
videos, TAICHI frames, …) drop into the same `data/<dataset>/` directories —
see [Dataset Setup](#dataset-setup).

---

## Dataset Setup

All datasets live under `$DIFFSDA_DATASETS_ROOT` (default: `./data`).

For every dataset, files marked **[bundle]** ship inside the Google Drive
release tarball above (extract it and they appear in the right place
automatically). Files marked **[user]** must be obtained from the original
source — most are public, a few require registration.

### Video Datasets

#### MUG Facial Expression Database (`--dataset mug`)

License-restricted — request access at https://mug.ee.auth.gr/fed/.

```
$DIFFSDA_DATASETS_ROOT/
  mug_pre2_train/        # [user] preprocessed training frames *
  mug_pre2_test/         # [user] preprocessed test frames *
  MUG/
    subjects3/           # [user] per-subject directories (identity labels)
```

\* The `mug_pre2_*` directories are produced by the lab's preprocessing
pipeline (face crop + frame resampling). Because MUG has license restrictions
we **cannot redistribute them**. If you have a MUG license, contact the
maintainers for the preprocessing scripts, or run `--dataset mug_small` /
the small variants which only need the raw `subjects3/` structure.

#### TAICHI (`--dataset taichi`)

```
$DIFFSDA_DATASETS_ROOT/
  TAICHI/
    taichi-png/
      train/                          # [user] PNG frames per video clip
      test/                           # [user] PNG frames per video clip
      mean_std_256_vq8.npz            # [bundle] latent normalization stats
```

**Download (raw):** follow `taichi-png` instructions in https://github.com/AliaksandrSiarohin/first-order-model.

**Latent encoding** (only needed if you want `--latent_dataset`):
```bash
cd preprocessing
python taichi_to_latent.py --input_size 256 --first_stage_model vq8
# Slurm job array (10 sections):
sbatch latent_preprocessing.sh
```

#### VoxCeleb1 (`--dataset vox1`)

```
$DIFFSDA_DATASETS_ROOT/
  VoxCeleb/
    unzippedIntervalFaces/data/<idXXXXX>/...   # [user] cropped face JPEGs
    mean_std_256_vq4.npz                       # [bundle]
    mean_std_256_vq8.npz                       # [bundle]
    mean_std_256_vq8ft.npz                     # [bundle]
```

**Download (raw):** https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html — pick "Cropped face images" (~36 GB, requires free registration).

**Latent encoding:**
```bash
cd preprocessing
python vox1_to_latent_finetuned_vq.py --input_size 256 --first_stage_model vq8ft --section 1 --range 10
# run sections 1..10 in parallel under Slurm
```

#### CelebV-HQ (`--dataset celebv`)

```
$DIFFSDA_DATASETS_ROOT/
  CelebV-HQ/
    35666/                                     # [user] mp4 clips
    celebvhq_info.json                         # [user] metadata
    filtered_clips.pkl                         # [bundle] preprocessing input
    filtered_clips_new.pkl                     # [bundle] train split index
    filtered_clips_new_test.pkl                # [bundle] test split index
    mean_std_256_vq4.npz                       # [bundle]
    mean_std_256_vq8.npz                       # [bundle]
    mean_std_256_vq8ft.npz                     # [bundle]
```

**Download (raw):** https://celebv-hq.github.io/ — mp4 clips + `celebvhq_info.json`.

**Preprocessing:**
```bash
cd preprocessing
python celebv_to_crop.py --input_size 512 --section 1            # face crop
python celebv_to_latent.py --input_size 256 --first_stage_model vq8ft --face_crop --section 1
```

---

### Audio / Speech Datasets

#### TIMIT (`--dataset timit`)

**Download:** LDC93S1 — https://catalog.ldc.upenn.edu/LDC93S1.

License-restricted (LDC93S1) — purchase / institutional access at https://catalog.ldc.upenn.edu/LDC93S1.

```
$DIFFSDA_DATASETS_ROOT/
  TIMIT/
    TIMIT/data/{TRAIN,TEST}/...        # [user] raw waveforms
```

Phone/word alignment `.json` files are checked into `timit_annotations/` —
already in the repo, no extra download needed.

#### LibriSpeech (`--dataset libri`)

```
$DIFFSDA_LIBRI_ROOT/                   # default: $DIFFSDA_DATASETS_ROOT/LibriSpeech
  train-clean-360/                     # [user]
  dev-clean/                           # [user]
  test-clean/                          # [user]
  mean_std.pkl                         # [bundle] mel-spec mean/std
```

**Download (raw):** https://openslr.org/12/ — fetch `train-clean-360`,
`dev-clean`, `test-clean`.

If you do not download the bundle, regenerate `mean_std.pkl` once:
```python
from datasets_util.LibriSpeech import calc_mean_and_std
calc_mean_and_std()   # writes mean_std.pkl into DIFFSDA_LIBRI_ROOT
```

---

### Time-Series Datasets

#### Air Quality (`--dataset airq`)

```
$DIFFSDA_DATASETS_ROOT/
  air_quality/
    PRSA_Data_Aotizhongxin_20130301-20170228.csv
    ...   (12 station files)            # [user]
```

**Download:** https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data — public CSV download, no auxiliary files needed.

#### ETT-h (`--dataset etth`)

```
$DIFFSDA_DATASETS_ROOT/
  Ett_ICLR/
    ETTh1.csv                           # [user]
    ETTh2.csv                           # [user]
```

**Download:** https://github.com/zhouhaoyi/ETDataset — `ETTh1.csv` and
`ETTh2.csv`. No auxiliary files needed.

#### PhysioNet 2012 (`--dataset physionet`)

```
$DIFFSDA_DATASETS_ROOT/
  physionet/
    set-a/                              # [user] raw patient .txt files
    Outcomes-a.txt                      # [bundle]
    processed_df.csv                    # [bundle] (~5 min faster startup)
    processed_static_df.csv             # [bundle]
```

**Download (raw):** https://physionet.org/content/challenge-2012/1.0.0/ — fetch `set-a.tar.gz` and extract.

The `processed_*.csv` files are produced by the data loader on first run; the
bundle ships them so you skip a one-time ~5 min preprocessing pass.

---

## Training

The model (`--model timediffpriorkarras`) and dataset (`--dataset`) flags pick
the architecture and data loader. Multi-GPU training uses DDP automatically — set
`CUDA_VISIBLE_DEVICES` to control which GPUs are used.

### Video

```bash
python train_video.py \
    --mode train \
    --model timediffpriorkarras \
    --dataset vox1 \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --s_dim 32 --d_dim 4 \
    --hidden_dim 256 \
    --first_stage_model vq8ft \
    --latent_dataset
```

For other video datasets, replace `--dataset vox1` with `mug`, `taichi`, or `celebv`.

### Audio

```bash
# LibriSpeech
python train_audio.py \
    --mode train \
    --model timediffpriorkarras \
    --dataset libri \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --s_dim 16 --d_dim 8 \
    --hidden_dim 256 \
    --mlp_hidden_dim 128 \
    --mlp_hidden_dim_enc 128 \
    --mel

# TIMIT
python train_audio.py \
    --mode train \
    --model timediffpriorkarras \
    --dataset timit \
    --epochs 1000 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --s_dim 16 --d_dim 8 \
    --hidden_dim 256 \
    --mel
```

### Time-Series

```bash
# PhysioNet
python train_timeseries.py \
    --mode train --r_seed 42 \
    --dataset physionet \
    --model timediffpriorkarras \
    --s_dim 24 --d_dim 2 \
    --hidden_dim 96 \
    --diffusion_steps 24 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --mlp_hidden_dim 256 \
    --mlp_hidden_dim_enc 96 \
    --ch_mult 1 2 2 2

# Air Quality
python train_timeseries.py \
    --mode train --r_seed 42 \
    --dataset airq \
    --model timediffpriorkarras \
    --s_dim 16 --d_dim 4 \
    --hidden_dim 512 \
    --diffusion_steps 16 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --mlp_hidden_dim 256 \
    --mlp_hidden_dim_enc 128 \
    --ch_mult 1 2 2 2

# ETT-h
python train_timeseries.py \
    --mode train --r_seed 42 \
    --dataset etth \
    --model timediffpriorkarras \
    --s_dim 16 --d_dim 4 \
    --hidden_dim 512 \
    --diffusion_steps 32 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --mlp_hidden_dim 128 \
    --mlp_hidden_dim_enc 256 \
    --ch_mult 1 2 2 2
```

Trained checkpoints land in `$DIFFSDA_MODELS_ROOT/<exp-string>/model-<epoch>.pth`.

---

## Evaluation

All commands below assume the released checkpoints are downloaded to
`$DIFFSDA_FINAL_WEIGHTS` (default: `./checkpoints/DiffSDA/`) and that the
matching raw datasets are present under `$DIFFSDA_DATASETS_ROOT`. Every script
prints metrics to stdout and writes a one-line summary under `./results/`.

### Video — Reconstruction AKD / AED (paper Table 3)

`exp_aed_akd_faces.py` covers face datasets (uses `face_alignment` for AKD and
`DeepFace.verify(model_name='VGG-Face', distance_metric='euclidean')` for AED):

```bash
python -m evaluation.faces.exp_aed_akd_faces --dataset vox1   --mode eval
python -m evaluation.faces.exp_aed_akd_faces --dataset celebv --mode eval
python -m evaluation.faces.exp_aed_akd_faces --dataset mug    --mode eval
```

TaiChi-HD uses body-pose + person re-id (OpenPose / Layumi-reid) instead of face
metrics.

**1. Pin the submodules at the expected commits.** Other commits change layer
shapes / import paths and the released TaiChi checkpoint will not load.

```bash
git submodule update --init OpenFacePytorch pose_estimation reid_baseline
git -C OpenFacePytorch checkout 548149a
git -C pose_estimation checkout 257d8b4
git -C reid_baseline   checkout 2a5a656
```

**2. Download the pose + re-id weights.** `OpenFacePytorch` ships its weights
inside the submodule; the other two do not:

```bash
# OpenPose body-pose model -> pose_estimation/network/weight/pose_model.pth
mkdir -p pose_estimation/network/weight
wget -O pose_estimation/network/weight/pose_model.pth \
    'https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=1'

# Person re-id model -> reid_baseline/reid_model.pth
#   GoogleDrive id from layumi/Person_reID_baseline_pytorch:
#   https://drive.google.com/open?id=1__x0qNJ3T654wTghmuRjydn42NsAZW_M
# Download the file manually or via gdown, then move it into place:
gdown --id 1__x0qNJ3T654wTghmuRjydn42NsAZW_M -O reid_baseline/reid_model.pth
```

If you keep the weights elsewhere, point at them via env:

```bash
export DIFFSDA_POSE_WEIGHTS=/path/to/pose_model.pth
export DIFFSDA_REID_WEIGHTS=/path/to/reid_model.pth
```

**3. Run the evaluation:**

```bash
python -m evaluation.taichi.exp_taichi
```

### Video — Conditional Swap AKD / AED (paper Table 4)

```bash
python -m evaluation.faces.exp_aed_akd_faces --dataset vox1   --mode swap
python -m evaluation.faces.exp_aed_akd_faces --dataset celebv --mode swap
python -m evaluation.faces.exp_aed_akd_faces --dataset mug    --mode swap
python -m evaluation.taichi.exp_taichi_swap
```

Static-swap freezes the content code and replaces dynamics (measured by AED);
dynamic-swap is the reverse (measured by AKD).

### Video — Cross-Dataset Zero-Shot (paper Table 5)

```bash
python -m evaluation.faces.exp_aed_akd_faces --dataset vox1 --target mug --mode zeroshot
```

`--dataset` is the source model, `--target` is the dataset to evaluate on.

### Audio — Speaker Verification EER (paper Table 5)

Static EER (identity preservation) and Dynamic EER (content preservation). The
LibriSpeech run is force-batched to 1 to match the original benchmark; TIMIT
batch size is tunable via `DIFFSDA_TIMIT_EVAL_BATCH_SIZE`.

```bash
# LibriSpeech — reuse the same args as training
python -m evaluation.audio.exp_libri \
    --mode eval --model timediffpriorkarras --dataset libri \
    --s_dim 16 --d_dim 8 --hidden_dim 256 \
    --mlp_hidden_dim 128 --mlp_hidden_dim_enc 128 --mel

# TIMIT
python -m evaluation.audio.exp_timit_eer \
    --mode eval --model timediffpriorkarras --dataset timit \
    --s_dim 16 --d_dim 8 --hidden_dim 256 --mel
```

### Time-Series (paper Table 2)

`exp_ts.py` reports AUPRC/AUROC (PhysioNet, mortality predictor), classification
accuracy (AirQ), or MAE (ETTh, oil-temperature predictor) — all averaged across
multiple seeds (set `DIFFSDA_TS_CLS_CV` to override the default 2):

```bash
python -m evaluation.ts.exp_ts --dataset physionet
python -m evaluation.ts.exp_ts --dataset airq
python -m evaluation.ts.exp_ts --dataset etth
```

**Discriminative score** (post-hoc discriminator between real and generated
sequences):

```bash
python -m evaluation.ts.discriminative_torch
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zisling2026diffsda,
    title={DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities},
    author={Zisling, Hedi and Naiman, Ilan and Berman, Nimrod and Suwajanakorn, Supasorn and Azencot, Omri},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026},
    url={https://openreview.net/forum?id=tooDJHBSvO}
}
```
