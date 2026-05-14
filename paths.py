"""Centralized path configuration.

All dataset, model, and checkpoint locations are resolved from environment
variables with sensible defaults that work out-of-the-box from a fresh clone.

Defaults
--------
``DIFFSDA_ROOT`` is the project root (the directory of this file). The remaining
defaults are derived from it:

    datasets   -> $DIFFSDA_ROOT/data
    weights    -> $DIFFSDA_ROOT/checkpoints
    samples    -> $DIFFSDA_ROOT/samples
    logs       -> $DIFFSDA_ROOT/logs

Set any of the environment variables below in your shell to override.

Environment variables
---------------------
DIFFSDA_ROOT             repo / project root (default: this file's parent)
DIFFSDA_DATASETS_ROOT    where raw / preprocessed datasets live
DIFFSDA_MODELS_ROOT      where trained model checkpoints are saved
DIFFSDA_SAMPLES_ROOT     where generated samples are written
DIFFSDA_PRETRAINED_ROOT  where pre-trained VQ / aux weights live
DIFFSDA_FINAL_WEIGHTS    where released DiffSDA model weights live
DIFFSDA_CLASSIFIERS_ROOT where evaluation classifiers live
DIFFSDA_LIBRI_ROOT       root of the LibriSpeech corpus
DIFFSDA_LOGS_ROOT        where Slurm / training logs are written
"""

import os


def _env(key, default):
    v = os.environ.get(key)
    return v if v else default


_HERE = os.path.dirname(os.path.abspath(__file__))

DIFFSDA_ROOT = _env('DIFFSDA_ROOT', _HERE)

DATASETS_ROOT = _env('DIFFSDA_DATASETS_ROOT',
                     os.path.join(DIFFSDA_ROOT, 'data'))
MODELS_ROOT = _env('DIFFSDA_MODELS_ROOT',
                   os.path.join(DIFFSDA_ROOT, 'checkpoints', 'runs'))
SAMPLES_ROOT = _env('DIFFSDA_SAMPLES_ROOT',
                    os.path.join(DIFFSDA_ROOT, 'samples'))
LOGS_ROOT = _env('DIFFSDA_LOGS_ROOT',
                 os.path.join(DIFFSDA_ROOT, 'logs'))

# VQ / autoencoder pre-trained weights (LDM checkpoints)
PRETRAINED_ROOT = _env('DIFFSDA_PRETRAINED_ROOT',
                       os.path.join(DIFFSDA_ROOT, 'checkpoints', 'vq_models'))

# Released DiffSDA model checkpoints (one .pth per dataset)
FINAL_WEIGHTS_ROOT = _env('DIFFSDA_FINAL_WEIGHTS',
                          os.path.join(DIFFSDA_ROOT, 'checkpoints', 'DiffSDA'))

# Pre-trained evaluation classifiers (e.g. MUG action classifier)
CLASSIFIERS_ROOT = _env('DIFFSDA_CLASSIFIERS_ROOT',
                        os.path.join(DIFFSDA_ROOT, 'checkpoints', 'classifiers'))

# Per-dataset subpaths, relative to DATASETS_ROOT.
DATASET_DIRS = {
    'vox1':        os.path.join(DATASETS_ROOT, 'VoxCeleb', 'unzippedIntervalFaces'),
    'celebv':      os.path.join(DATASETS_ROOT, 'CelebV-HQ'),
    'taichi':      os.path.join(DATASETS_ROOT, 'TAICHI', 'taichi-png'),
    'mug_train':   os.path.join(DATASETS_ROOT, 'mug_pre2_train'),
    'mug_test':    os.path.join(DATASETS_ROOT, 'mug_pre2_test'),
    'mug_subjects': os.path.join(DATASETS_ROOT, 'MUG', 'subjects3'),
    'mug_root':    os.path.join(DATASETS_ROOT, 'MUG'),
    'timit':       os.path.join(DATASETS_ROOT, 'TIMIT', 'TIMIT'),
    'timit_annotations': os.path.join(_HERE, 'timit_annotations'),
    'airq':        os.path.join(DATASETS_ROOT, 'air_quality'),
    'etth':        os.path.join(DATASETS_ROOT, 'Ett_ICLR'),
    'physionet':   os.path.join(DATASETS_ROOT, 'physionet'),
}

# LibriSpeech often lives on a separate volume; keep an independent override.
LIBRI_ROOT = _env('DIFFSDA_LIBRI_ROOT',
                  os.path.join(DATASETS_ROOT, 'LibriSpeech'))

# Pair-split CSV files committed as project assets in splits/
SPLITS_ROOT = os.path.join(_HERE, 'splits')
