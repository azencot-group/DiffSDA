"""
Data preparation.

Download: https://catalog.ldc.upenn.edu/LDC93S1

Authors
* Mirco Ravanelli 2020
* Elena Rastorgueva 2020
"""

import os
import json
import logging
import speechbrain as sb
import torch
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio
import torch as th
import itertools
import scipy

from HiFiGAN import audio_path_to_data_hifi
from datasets_util.LibriSpeech import libri_normalize

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_timit(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    phn_set=39,
    uppercase=False,
    skip_prep=False,
):
    """
    repares the json files for the TIMIT dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original TIMIT dataset is stored.
    save_json_train : str
        The path where to store the training json file.
    save_json_valid : str
        The path where to store the valid json file.
    save_json_test : str
        The path where to store the test json file.
    phn_set : {60, 48, 39}, optional,
        Default: 39
        The phoneme set to use in the phn label.
        It could be composed of 60, 48, or 39 phonemes.
    uppercase : bool, optional
        Default: False
        This option must be True when the TIMIT dataset
        is in the upper-case version.
    skip_prep: bool
        Default: False
        If True, the data preparation is skipped.

    Example
    -------
    # >>> from recipes.TIMIT.timit_prepare import prepare_timit
    # >>> data_folder = 'datasets/TIMIT'
    # >>> prepare_timit(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Skip if needed
    if skip_prep:
        return

    # Getting speaker dictionary
    dev_spk, test_spk = _get_speaker()
    avoid_sentences = ["sa1", "sa2"]
    extension = [".wav"]

    # Checking TIMIT_uppercase
    if uppercase:
        avoid_sentences = [item.upper() for item in avoid_sentences]
        extension = [item.upper() for item in extension]
        dev_spk = [item.upper() for item in dev_spk]
        test_spk = [item.upper() for item in test_spk]

    # Check if this phase is already done (if so, skip it)
    if skip([save_json_train, save_json_valid, save_json_test]):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains TIMIT
    _check_timit_folders(uppercase, data_folder)

    msg = "Creating json files for the TIMIT Dataset.."
    logger.info(msg)

    # Creating json files
    # NOTE: TIMIT has the DEV files in the test directory.
    splits = ["train", "test", "test"]
    annotations = [save_json_train, save_json_valid, save_json_test]
    match_or = [None, dev_spk, test_spk]

    for split, save_file, match in zip(splits, annotations, match_or):
        if uppercase:
            match_lst = extension + [split.upper()]
        else:
            match_lst = extension + [split]

        # List of the wav files
        wav_lst = get_all_files(
            data_folder,
            match_and=match_lst,
            match_or=match,
            exclude_or=avoid_sentences,
        )
        if split == "dev":
            print(wav_lst)

        # Json creation
        create_json(wav_lst, save_file, uppercase, phn_set)


def _get_phonemes():

    # This dictionary is used to conver the 60 phoneme set
    # into the 48 one
    from_60_to_48_phn = {}
    from_60_to_48_phn["sil"] = "sil"
    from_60_to_48_phn["aa"] = "aa"
    from_60_to_48_phn["ae"] = "ae"
    from_60_to_48_phn["ah"] = "ah"
    from_60_to_48_phn["ao"] = "ao"
    from_60_to_48_phn["aw"] = "aw"
    from_60_to_48_phn["ax"] = "ax"
    from_60_to_48_phn["ax-h"] = "ax"
    from_60_to_48_phn["axr"] = "er"
    from_60_to_48_phn["ay"] = "ay"
    from_60_to_48_phn["b"] = "b"
    from_60_to_48_phn["bcl"] = "vcl"
    from_60_to_48_phn["ch"] = "ch"
    from_60_to_48_phn["d"] = "d"
    from_60_to_48_phn["dcl"] = "vcl"
    from_60_to_48_phn["dh"] = "dh"
    from_60_to_48_phn["dx"] = "dx"
    from_60_to_48_phn["eh"] = "eh"
    from_60_to_48_phn["el"] = "el"
    from_60_to_48_phn["em"] = "m"
    from_60_to_48_phn["en"] = "en"
    from_60_to_48_phn["eng"] = "ng"
    from_60_to_48_phn["epi"] = "epi"
    from_60_to_48_phn["er"] = "er"
    from_60_to_48_phn["ey"] = "ey"
    from_60_to_48_phn["f"] = "f"
    from_60_to_48_phn["g"] = "g"
    from_60_to_48_phn["gcl"] = "vcl"
    from_60_to_48_phn["h#"] = "sil"
    from_60_to_48_phn["hh"] = "hh"
    from_60_to_48_phn["hv"] = "hh"
    from_60_to_48_phn["ih"] = "ih"
    from_60_to_48_phn["ix"] = "ix"
    from_60_to_48_phn["iy"] = "iy"
    from_60_to_48_phn["jh"] = "jh"
    from_60_to_48_phn["k"] = "k"
    from_60_to_48_phn["kcl"] = "cl"
    from_60_to_48_phn["l"] = "l"
    from_60_to_48_phn["m"] = "m"
    from_60_to_48_phn["n"] = "n"
    from_60_to_48_phn["ng"] = "ng"
    from_60_to_48_phn["nx"] = "n"
    from_60_to_48_phn["ow"] = "ow"
    from_60_to_48_phn["oy"] = "oy"
    from_60_to_48_phn["p"] = "p"
    from_60_to_48_phn["pau"] = "sil"
    from_60_to_48_phn["pcl"] = "cl"
    from_60_to_48_phn["q"] = ""
    from_60_to_48_phn["r"] = "r"
    from_60_to_48_phn["s"] = "s"
    from_60_to_48_phn["sh"] = "sh"
    from_60_to_48_phn["t"] = "t"
    from_60_to_48_phn["tcl"] = "cl"
    from_60_to_48_phn["th"] = "th"
    from_60_to_48_phn["uh"] = "uh"
    from_60_to_48_phn["uw"] = "uw"
    from_60_to_48_phn["ux"] = "uw"
    from_60_to_48_phn["v"] = "v"
    from_60_to_48_phn["w"] = "w"
    from_60_to_48_phn["y"] = "y"
    from_60_to_48_phn["z"] = "z"
    from_60_to_48_phn["zh"] = "zh"

    # This dictionary is used to conver the 60 phoneme set
    from_60_to_39_phn = {}
    from_60_to_39_phn["sil"] = "sil"
    from_60_to_39_phn["aa"] = "aa"
    from_60_to_39_phn["ae"] = "ae"
    from_60_to_39_phn["ah"] = "ah"
    from_60_to_39_phn["ao"] = "aa"
    from_60_to_39_phn["aw"] = "aw"
    from_60_to_39_phn["ax"] = "ah"
    from_60_to_39_phn["ax-h"] = "ah"
    from_60_to_39_phn["axr"] = "er"
    from_60_to_39_phn["ay"] = "ay"
    from_60_to_39_phn["b"] = "b"
    from_60_to_39_phn["bcl"] = "sil"
    from_60_to_39_phn["ch"] = "ch"
    from_60_to_39_phn["d"] = "d"
    from_60_to_39_phn["dcl"] = "sil"
    from_60_to_39_phn["dh"] = "dh"
    from_60_to_39_phn["dx"] = "dx"
    from_60_to_39_phn["eh"] = "eh"
    from_60_to_39_phn["el"] = "l"
    from_60_to_39_phn["em"] = "m"
    from_60_to_39_phn["en"] = "n"
    from_60_to_39_phn["eng"] = "ng"
    from_60_to_39_phn["epi"] = "sil"
    from_60_to_39_phn["er"] = "er"
    from_60_to_39_phn["ey"] = "ey"
    from_60_to_39_phn["f"] = "f"
    from_60_to_39_phn["g"] = "g"
    from_60_to_39_phn["gcl"] = "sil"
    from_60_to_39_phn["h#"] = "sil"
    from_60_to_39_phn["hh"] = "hh"
    from_60_to_39_phn["hv"] = "hh"
    from_60_to_39_phn["ih"] = "ih"
    from_60_to_39_phn["ix"] = "ih"
    from_60_to_39_phn["iy"] = "iy"
    from_60_to_39_phn["jh"] = "jh"
    from_60_to_39_phn["k"] = "k"
    from_60_to_39_phn["kcl"] = "sil"
    from_60_to_39_phn["l"] = "l"
    from_60_to_39_phn["m"] = "m"
    from_60_to_39_phn["ng"] = "ng"
    from_60_to_39_phn["n"] = "n"
    from_60_to_39_phn["nx"] = "n"
    from_60_to_39_phn["ow"] = "ow"
    from_60_to_39_phn["oy"] = "oy"
    from_60_to_39_phn["p"] = "p"
    from_60_to_39_phn["pau"] = "sil"
    from_60_to_39_phn["pcl"] = "sil"
    from_60_to_39_phn["q"] = ""
    from_60_to_39_phn["r"] = "r"
    from_60_to_39_phn["s"] = "s"
    from_60_to_39_phn["sh"] = "sh"
    from_60_to_39_phn["t"] = "t"
    from_60_to_39_phn["tcl"] = "sil"
    from_60_to_39_phn["th"] = "th"
    from_60_to_39_phn["uh"] = "uh"
    from_60_to_39_phn["uw"] = "uw"
    from_60_to_39_phn["ux"] = "uw"
    from_60_to_39_phn["v"] = "v"
    from_60_to_39_phn["w"] = "w"
    from_60_to_39_phn["y"] = "y"
    from_60_to_39_phn["z"] = "z"
    from_60_to_39_phn["zh"] = "sh"

    return from_60_to_48_phn, from_60_to_39_phn


def _get_speaker():

    # List of test speakers
    test_spk = [
        "fdhc0",
        "felc0",
        "fjlm0",
        "fmgd0",
        "fmld0",
        "fnlp0",
        "fpas0",
        "fpkt0",
        "mbpm0",
        "mcmj0",
        "mdab0",
        "mgrt0",
        "mjdh0",
        "mjln0",
        "mjmp0",
        "mklt0",
        "mlll0",
        "mlnt0",
        "mnjm0",
        "mpam0",
        "mtas1",
        "mtls0",
        "mwbt0",
        "mwew0",
    ]

    # List of dev speakers
    dev_spk = [
        "fadg0",
        "faks0",
        "fcal1",
        "fcmh0",
        "fdac1",
        "fdms0",
        "fdrw0",
        "fedw0",
        "fgjd0",
        "fjem0",
        "fjmg0",
        "fjsj0",
        "fkms0",
        "fmah0",
        "fmml0",
        "fnmr0",
        "frew0",
        "fsem0",
        "majc0",
        "mbdg0",
        "mbns0",
        "mbwm0",
        "mcsh0",
        "mdlf0",
        "mdls0",
        "mdvc0",
        "mers0",
        "mgjf0",
        "mglb0",
        "mgwt0",
        "mjar0",
        "mjfc0",
        "mjsw0",
        "mmdb1",
        "mmdm2",
        "mmjr0",
        "mmwh0",
        "mpdf0",
        "mrcs0",
        "mreb0",
        "mrjm4",
        "mrjr0",
        "mroa0",
        "mrtk0",
        "mrws1",
        "mtaa0",
        "mtdt0",
        "mteb0",
        "mthc0",
        "mwjg0",
    ]

    return dev_spk, test_spk


def skip(annotations):
    """
    Detects if the timit data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    skip = True

    for annotation in annotations:
        if not os.path.isfile(annotation):
            skip = False
            break

    return skip


def create_json(
    wav_lst, json_file, uppercase, phn_set,
):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    json_file : str
            The path of the output json file.
    uppercase : bool
        Whether this is the uppercase version of timit.
    phn_set : {60, 48, 39}, optional,
        Default: 39
        The phoneme set to use in the phn label.
    """

    # Adding some Prints
    msg = "Creating %s..." % (json_file)
    logger.info(msg)
    json_dict = {}

    for wav_file in wav_lst:

        # Getting sentence and speaker ids
        spk_id = wav_file.split("/")[-2]
        snt_id = wav_file.split("/")[-1].replace(".wav", "")
        snt_id = spk_id + "_" + snt_id

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = len(signal) / SAMPLERATE

        # Retrieving words and check for uppercase
        if uppercase:
            wrd_file = wav_file.replace(".WAV", ".WRD")
        else:
            wrd_file = wav_file.replace(".wav", ".wrd")

        if not os.path.exists(os.path.dirname(wrd_file)):
            err_msg = "the wrd file %s does not exists!" % (wrd_file)
            raise FileNotFoundError(err_msg)

        words = [line.rstrip("\n").split(" ")[2] for line in open(wrd_file)]
        words = " ".join(words)

        # Retrieving phonemes
        if uppercase:
            phn_file = wav_file.replace(".WAV", ".PHN")
        else:
            phn_file = wav_file.replace(".wav", ".phn")

        if not os.path.exists(os.path.dirname(phn_file)):
            err_msg = "the wrd file %s does not exists!" % (phn_file)
            raise FileNotFoundError(err_msg)

        # Getting the phoneme and ground truth ends lists from the phn files
        phonemes, ends = get_phoneme_lists(phn_file, phn_set)

        json_dict[snt_id] = {
            "wav": wav_file,
            "duration": duration,
            "spk_id": spk_id,
            "phn": phonemes,
            "wrd": words,
            "ground_truth_phn_ends": ends,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def get_phoneme_lists(phn_file, phn_set):
    """
    Reads the phn file and gets the phoneme list & ground truth ends list.
    """

    phonemes = []
    ends = []

    for line in open(phn_file):
        end, phoneme = line.rstrip("\n").replace("h#", "sil").split(" ")[1:]

        # Getting dictionaries for phoneme conversion
        from_60_to_48_phn, from_60_to_39_phn = _get_phonemes()

        # Removing end corresponding to q if phn set is not 61
        if phn_set != 60:
            if phoneme == "q":
                end = ""

        # Converting phns if necessary
        if phn_set == 48:
            phoneme = from_60_to_48_phn[phoneme]
        if phn_set == 39:
            phoneme = from_60_to_39_phn[phoneme]

        # Appending arrays
        if len(phoneme) > 0:
            phonemes.append(phoneme)
        if len(end) > 0:
            ends.append(end)

    if phn_set != 60:
        # Filtering out consecutive silences by applying a mask with `True` marking
        # which sils to remove
        # e.g.
        # phonemes          [  "a", "sil", "sil",  "sil",   "b"]
        # ends              [   1 ,    2 ,    3 ,     4 ,    5 ]
        # ---
        # create:
        # remove_sil_mask   [False,  True,  True,  False,  False]
        # ---
        # so end result is:
        # phonemes ["a", "sil", "b"]
        # ends     [  1,     4,   5]

        remove_sil_mask = [True if x == "sil" else False for x in phonemes]

        for i, val in enumerate(remove_sil_mask):
            if val is True:
                if i == len(remove_sil_mask) - 1:
                    remove_sil_mask[i] = False
                elif remove_sil_mask[i + 1] is False:
                    remove_sil_mask[i] = False

        phonemes = [
            phon for i, phon in enumerate(phonemes) if not remove_sil_mask[i]
        ]
        ends = [end for i, end in enumerate(ends) if not remove_sil_mask[i]]

    # Convert to e.g. "a sil b", "1 4 5"
    phonemes = " ".join(phonemes)
    ends = " ".join(ends)

    return phonemes, ends


def _check_timit_folders(uppercase, data_folder):
    """
    Check if the data folder actually contains the TIMIT dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain TIMIT dataset.
    """

    # Creating checking string wrt to lower or uppercase
    if uppercase:
        test_str = "/TEST/DR1"
        train_str = "/TRAIN/DR1"
    else:
        test_str = "/test/dr1"
        train_str = "/train/dr1"

    # Checking test/dr1
    if not os.path.exists(data_folder + test_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the TIMIT dataset)" % (data_folder + test_str)
        )
        raise FileNotFoundError(err_msg)

    # Checking train/dr1
    if not os.path.exists(data_folder + train_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the TIMIT dataset)" % (data_folder + train_str)
        )
        raise FileNotFoundError(err_msg)


def audio_path_to_data(X):
    """The pipline:
        1. we import the audio wav file in size (t x 1)
        2. we chunk the audio to 200ms frames = size(3200 x 1)
        3. we pad all frames shorter then 200ms with zeros
        4. we chunk it into a new batch = (new_batch_size x 3200)
        5. """
    # load audio
    # audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    sampling_rate = 14
    audio_list = []
    chunck_nums = []
    for x in X:
        full_audio = sb.dataio.dataio.read_audio(x)
        audio_chunks = list(th.split(full_audio, 3200*sampling_rate))
        audio_chunks[-1] = th.concat((audio_chunks[-1], th.zeros(3200*sampling_rate - audio_chunks[-1].shape[0])))
        audio_list = audio_list + audio_chunks
        chunck_nums.append(len(audio_chunks))

    return th.stack(audio_list, dim=0), chunck_nums


def audio_path_to_data2(X):
    """The pipline:
        1. we import the audio wav file in size (t x 1)
        2. we chunk the audio to 200ms frames = size(3200 x 1)
        3. we pad all frames shorter then 200ms with zeros
        4. we chunk it into a new batch = (new_batch_size x 3200)
        5. """
    # load audio
    audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    max_len = max([len(x) for x in audio_list])

    # pad signals
    padded_list = [th.concat((x, th.zeros(max_len - x.shape[0]))) for x in audio_list]

    return th.stack(padded_list, dim=0), [len(x) // 165 for x in audio_list]




def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # set the data folder
    data_folder = hparams.data_folder
    # get train annotations
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams.train_annotation,
        replacements={"data_root": data_folder},
    )

    # we sort training data to speed up training and get better results.
    train_data = train_data.filtered_sorted(sort_key="duration")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams.valid_annotation,
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams.test_annotation,
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = th.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = th.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = th.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    return train_data, valid_data, test_data, label_encoder


@th.no_grad()
def voice_verification_mean(args, model, spectrogram, test_loader, resampler):
    for epoch in range(5 if args.dataset == 'timit' else 1):

        print("Epoch", epoch)
        model.eval()
        # mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        # mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        # pred1_all, pred2_all, label2_all = list(), list(), list()
        # label_gt = list()

        dataset = list(test_loader.dataset)
        if args.mel:
            wavs, chunck_nums = audio_path_to_data_hifi([w['wav'] for w in dataset], resampler)
        else:
            wavs, chunck_nums = audio_path_to_data([w['wav'] for w in dataset])

        # X = wavs
        # # encode
        # X = X.cuda()
        # X = spectrogram(X)
        # X = X.permute(0, 2, 1)
        data = wavs
        s_mean_lst, d_mean_lst = [], []
        i = 0
        while i < data.shape[0]:
            x = data[i:(i + args.batch_size)]
            x = x.to(device=args.device)
            x = spectrogram(x)
            x = x.permute(0, 2, 1)
            if args.dataset in ['timit']:
                x = timit_normalize(args, x)
            elif args.dataset in ['libri']:
                x = libri_normalize(args, x)
            s, d, a = model.encoder(x.squeeze(dim=1))
            s_mean_lst.append(s.cpu())
            d_mean_lst.append(d.cpu())
            i = i + args.batch_size

        s_mean = th.vstack(s_mean_lst)
        d_mean = th.vstack(d_mean_lst)

        n = 0
        f_means_all = []
        for num in chunck_nums:
            f_means_all.append(th.mean(s_mean[n:n + num], dim=0))
            n += num
        f_post_mean = th.stack(f_means_all)

        n = 0
        z_means_all = []
        for num in chunck_nums:
            z_means_all.append(th.mean(th.mean(d_mean[n:n + num], dim=0), dim=0))
            n += num
        z_post_mean = th.stack(z_means_all)

        # --- create pairs of verifications and expected output ---
        index_comb = list(itertools.combinations(range(len(dataset)), 2))

        static_pairs = []
        dynamic_pairs = []
        dataset_list = list(dataset)
        # take the frames for each pair of samples
        for comb in index_comb:
            static_pairs.append(
                [f_post_mean[comb[0]], f_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])
            dynamic_pairs.append(
                [z_post_mean[comb[0]], z_post_mean[comb[1]],
                 dataset_list[comb[0]]['spk_id'] == dataset_list[comb[1]]['spk_id']])

        # --- def for binary search ---
        def f_s(epsilon):
            f_p_static = 0
            f_n_static = 0
            positive = 0
            negative = 0
            for pair in static_pairs:
                decision = th.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_static = f_p_static + 1
                elif not decision and pair[2]:
                    f_n_static = f_n_static + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_static / positive) - (f_p_static / negative)

        def f_d(epsilon):
            f_p_dyn = 0
            f_n_dyn = 0
            positive = 0
            negative = 0
            for pair in dynamic_pairs:
                decision = th.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > epsilon
                if decision and not pair[2]:
                    f_p_dyn = f_p_dyn + 1
                elif not decision and pair[2]:
                    f_n_dyn = f_n_dyn + 1

                if pair[2]:
                    positive = positive + 1
                else:
                    negative = negative + 1

            return (f_n_dyn / positive) - (f_p_dyn / negative)

        eps_static = scipy.optimize.bisect(f_s, -1, 1)
        eps_dynamic = scipy.optimize.bisect(f_d, -1, 1)

        # --- calculate the eer for dynamic and static ---
        f_p_static = 0
        f_n_static = 0
        score_static = 0
        positive = 0
        negative = 0
        for pair in static_pairs:
            decision = th.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_static
            if (decision and pair[2]) or (not decision and not pair[2]):
                score_static = score_static + 1
            elif decision and not pair[2]:
                f_p_static = f_p_static + 1
            elif not decision and pair[2]:
                f_n_static = f_n_static + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_static = (f_p_static) / negative

        f_p_dyn = 0
        f_n_dyn = 0
        positive = 0
        negative = 0
        for pair in dynamic_pairs:
            decision = th.cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)) > eps_dynamic
            if decision and not pair[2]:
                f_p_dyn = f_p_dyn + 1
            elif not decision and pair[2]:
                f_n_dyn = f_n_dyn + 1

            if pair[2]:
                positive = positive + 1
            else:
                negative = negative + 1

        eer_dynamic = (f_p_dyn) / negative

        return eer_static, eer_dynamic


timit_mean = torch.tensor([0.0087, 0.0165, 0.0409, 0.0682, 0.0923, 0.1245, 0.1463, 0.1467, 0.1550,
        0.1875, 0.2275, 0.2542, 0.2637, 0.2618, 0.2546, 0.2456, 0.2331, 0.2143,
        0.1928, 0.1714, 0.1515, 0.1344, 0.1212, 0.1110, 0.1033, 0.0967, 0.0906,
        0.0854, 0.0811, 0.0776, 0.0745, 0.0709, 0.0671, 0.0642, 0.0619, 0.0599,
        0.0584, 0.0571, 0.0557, 0.0543, 0.0529, 0.0513, 0.0494, 0.0473, 0.0451,
        0.0431, 0.0411, 0.0391, 0.0372, 0.0355, 0.0342, 0.0330, 0.0318, 0.0310,
        0.0304, 0.0302, 0.0301, 0.0303, 0.0306, 0.0309, 0.0311, 0.0312, 0.0311,
        0.0307, 0.0301, 0.0294, 0.0284, 0.0273, 0.0261, 0.0251, 0.0241, 0.0233,
        0.0227, 0.0222, 0.0219, 0.0217, 0.0217, 0.0218, 0.0220, 0.0222, 0.0225,
        0.0228, 0.0231, 0.0234, 0.0236, 0.0237, 0.0238, 0.0238, 0.0237, 0.0235,
        0.0232, 0.0229, 0.0225, 0.0221, 0.0216, 0.0211, 0.0207, 0.0202, 0.0198,
        0.0194, 0.0190, 0.0185, 0.0182, 0.0178, 0.0175, 0.0171, 0.0168, 0.0166,
        0.0163, 0.0161, 0.0159, 0.0157, 0.0155, 0.0152, 0.0150, 0.0147, 0.0144,
        0.0142, 0.0139, 0.0137, 0.0134, 0.0131, 0.0129, 0.0126, 0.0124, 0.0122,
        0.0120, 0.0118, 0.0116, 0.0114, 0.0113, 0.0111, 0.0110, 0.0108, 0.0106,
        0.0105, 0.0103, 0.0102, 0.0100, 0.0099, 0.0097, 0.0096, 0.0094, 0.0093,
        0.0091, 0.0090, 0.0089, 0.0088, 0.0087, 0.0086, 0.0086, 0.0085, 0.0085,
        0.0085, 0.0084, 0.0084, 0.0084, 0.0084, 0.0084, 0.0084, 0.0084, 0.0083,
        0.0082, 0.0082, 0.0081, 0.0081, 0.0080, 0.0079, 0.0079, 0.0078, 0.0078,
        0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0076,
        0.0076, 0.0076, 0.0076, 0.0076, 0.0076, 0.0076, 0.0076, 0.0076, 0.0076,
        0.0077, 0.0077, 0.0077, 0.0078, 0.0078, 0.0078, 0.0078, 0.0079, 0.0079,
        0.0079, 0.0078, 0.0072])

timit_std = torch.tensor([0.0261, 0.0310, 0.0592, 0.0931, 0.1171, 0.1572, 0.2048, 0.2328, 0.2553,
        0.3079, 0.3874, 0.4627, 0.5184, 0.5506, 0.5619, 0.5688, 0.5708, 0.5524,
        0.5140, 0.4664, 0.4176, 0.3734, 0.3354, 0.3107, 0.2956, 0.2809, 0.2636,
        0.2461, 0.2323, 0.2237, 0.2164, 0.2063, 0.1955, 0.1859, 0.1757, 0.1666,
        0.1604, 0.1558, 0.1508, 0.1446, 0.1403, 0.1365, 0.1307, 0.1240, 0.1182,
        0.1125, 0.1066, 0.1010, 0.0953, 0.0904, 0.0865, 0.0829, 0.0789, 0.0754,
        0.0733, 0.0722, 0.0719, 0.0720, 0.0725, 0.0728, 0.0732, 0.0732, 0.0728,
        0.0717, 0.0705, 0.0693, 0.0667, 0.0636, 0.0609, 0.0587, 0.0562, 0.0537,
        0.0520, 0.0508, 0.0502, 0.0502, 0.0505, 0.0510, 0.0515, 0.0518, 0.0525,
        0.0534, 0.0545, 0.0553, 0.0557, 0.0562, 0.0566, 0.0566, 0.0565, 0.0560,
        0.0551, 0.0546, 0.0541, 0.0534, 0.0527, 0.0519, 0.0509, 0.0501, 0.0495,
        0.0490, 0.0483, 0.0477, 0.0474, 0.0469, 0.0468, 0.0465, 0.0464, 0.0462,
        0.0458, 0.0457, 0.0456, 0.0454, 0.0453, 0.0450, 0.0445, 0.0439, 0.0437,
        0.0437, 0.0435, 0.0431, 0.0424, 0.0415, 0.0406, 0.0397, 0.0392, 0.0387,
        0.0380, 0.0373, 0.0369, 0.0366, 0.0363, 0.0362, 0.0357, 0.0349, 0.0345,
        0.0340, 0.0334, 0.0330, 0.0327, 0.0324, 0.0320, 0.0316, 0.0311, 0.0307,
        0.0304, 0.0300, 0.0296, 0.0291, 0.0288, 0.0287, 0.0285, 0.0285, 0.0287,
        0.0286, 0.0285, 0.0284, 0.0284, 0.0284, 0.0284, 0.0282, 0.0283, 0.0283,
        0.0281, 0.0278, 0.0275, 0.0272, 0.0269, 0.0267, 0.0266, 0.0265, 0.0263,
        0.0263, 0.0265, 0.0268, 0.0268, 0.0268, 0.0270, 0.0271, 0.0272, 0.0272,
        0.0271, 0.0270, 0.0271, 0.0272, 0.0272, 0.0273, 0.0272, 0.0273, 0.0277,
        0.0281, 0.0284, 0.0285, 0.0285, 0.0288, 0.0292, 0.0292, 0.0294, 0.0298,
        0.0299, 0.0300, 0.0304])


timit_mean_mel = torch.tensor([-8.1450, -7.6363, -6.9928, -6.7033, -6.5059, -6.3134, -6.3545, -6.4518,
        -6.3755, -6.2474, -6.0848, -6.1477, -6.1571, -6.2151, -6.2228, -6.3083,
        -6.3631, -6.4002, -6.4640, -6.5349, -6.5917, -6.5925, -6.7022, -6.7638,
        -6.8173, -6.8073, -6.9318, -6.9552, -6.9960, -7.0389, -7.0669, -7.0896,
        -7.1285, -7.1453, -7.2141, -7.1528, -7.2096, -7.1952, -7.1886, -7.2258,
        -7.2563, -7.2919, -7.3504, -7.3852, -7.4262, -7.4477, -7.4722, -7.4719,
        -7.4651, -7.4508, -7.4861, -7.5084, -7.5790, -7.6334, -7.6659, -7.6863,
        -7.6812, -7.6757, -7.6730, -7.6831, -7.7261, -7.7836, -7.8590, -7.9486,
        -8.0266, -8.1066, -8.1896, -8.2867, -8.3797, -8.4632, -8.5325, -8.5999,
        -8.6585, -8.6878, -8.7040, -8.7492, -8.8302, -8.9285, -9.0423, -9.1701])

timit_std_meal = torch.tensor([1.4439, 1.9567, 2.3782, 2.4025, 2.5160, 2.6227, 2.6177, 2.5834, 2.6449,
        2.7447, 2.8353, 2.8277, 2.8060, 2.8070, 2.7976, 2.7472, 2.7357, 2.7131,
        2.6724, 2.6292, 2.5914, 2.5592, 2.5099, 2.4911, 2.4688, 2.4531, 2.4338,
        2.4227, 2.4081, 2.3956, 2.3757, 2.3554, 2.3461, 2.3373, 2.3252, 2.3337,
        2.3202, 2.3175, 2.3131, 2.3024, 2.2865, 2.2615, 2.2369, 2.2086, 2.1865,
        2.1676, 2.1530, 2.1566, 2.1681, 2.1834, 2.1807, 2.1685, 2.1342, 2.1046,
        2.0905, 2.0889, 2.1018, 2.1171, 2.1255, 2.1224, 2.1020, 2.0750, 2.0442,
        2.0118, 1.9853, 1.9608, 1.9349, 1.9077, 1.8814, 1.8584, 1.8338, 1.8015,
        1.7688, 1.7471, 1.7305, 1.7046, 1.6745, 1.6495, 1.6250, 1.5983])

def timit_normalize(args, x):
    mean = timit_mean_mel if args.mel else timit_mean
    std = timit_std_meal if args.mel else timit_std
    return 0.5 * (x - mean.to(x.device)) / std.to(x.device)

def timit_denormalize(args, x):
    mean = timit_mean_mel if args.mel else timit_mean
    std = timit_std_meal if args.mel else timit_std
    return 2 * x * std.to(x.device) + mean.to(x.device)


