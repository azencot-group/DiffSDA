import numpy as np
import soundfile
import torch
# import matplotlib.pyplot as plt
# from scipy.io.wavfile import write
# import torchaudio
#
# from data import get_dataset_config, get_dataset
# from run import parse_args
# # from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F

MAX_WAV_VALUE = 32768.0
mel_basis = {}
hann_window = {}

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=22050,
                        hop_size=256, win_size=1024,
                        fmin=0, fmax=8000, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    fmax_key = f'{fmax}_{y.device}'
    if fmax_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft-hop_size)/2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann_window[str(y.device)], center=center,
                      pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = dynamic_range_compression(spec)  # spectral normalize
    return spec

def load_wav(full_path, torch_tensor=False):
    data, sampling_rate = soundfile.read(full_path, dtype='int16')
    if torch_tensor:
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    else:
        return data, sampling_rate


def audio_path_to_data_hifi(X, resampler, data_type='wav', sampling_rate=4):
    """The pipline:
        1. we import the audio wav file in size (t x 1)
        2. we chunk the audio to 200ms frames = size(3200 x 1)
        3. we pad all frames shorter then 200ms with zeros
        4. we chunk it into a new batch = (new_batch_size x 3200)
        5. """
    # load audio
    # audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    audio_list = []
    chunck_nums = []
    for x in X:
        full_audio, _ = load_wav(x, torch_tensor=True)
        audio_chunks = list(torch.split(full_audio, 3200*sampling_rate))
        audio_chunks[-1] = torch.concat((audio_chunks[-1], torch.zeros(3200*sampling_rate - audio_chunks[-1].shape[0])))
        audio_list = audio_list + audio_chunks
        chunck_nums.append(len(audio_chunks))
    audio = torch.stack(audio_list) / MAX_WAV_VALUE
    audio = resampler(audio)
    return audio, chunck_nums


def audio_path_to_data_hifi_full(X, resampler, data_type='wav', sampling_rate=4):
    # load audio
    # audio_list = [sb.dataio.dataio.read_audio(x) for x in X]
    audio_list = []
    chunck_nums = [1]
    for x in X:
        full_audio, _ = load_wav(x, torch_tensor=True)
        audio_chunks = list(torch.split(full_audio, 3200 * sampling_rate))
        audio_chunks[-1] = torch.concat((audio_chunks[-1], torch.zeros(3200 * sampling_rate - audio_chunks[-1].shape[0])))
        audio_list.append(torch.cat(audio_chunks))
        # audio_list.append(full_audio)
    audio = torch.stack(audio_list) / MAX_WAV_VALUE
    audio = resampler(audio)
    return audio, chunck_nums




# if __name__ == '__main__':
#     args = parse_args()
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     fastpitch, generator_train_setup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch')
#     hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
#     CHECKPOINT_SPECIFIC_ARGS = [
#         'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
#         'symbol_set', 'max_wav_value', 'prepend_space_to_text',
#         'append_space_to_text']
#
#     for k in CHECKPOINT_SPECIFIC_ARGS:
#         v1 = generator_train_setup.get(k, None)
#         v2 = vocoder_train_setup.get(k, None)
#
#         assert v1 is None or v2 is None or v1 == v2, \
#             f'{k} mismatch in spectrogram generator and vocoder'
#     fastpitch.to(device)
#     hifigan.to(device)
#     denoiser.to(device)
#     gen_kw = {'pace': 1.0,
#               'speaker': 0,
#               'pitch_tgt': None,
#               'pitch_transform': None}
#     denoising_strength = 0.005
#     hop_length = 256
#     win_length = 1024
#     sampling_rate = 22050
#     num_mels=80
#     f_max = 8000.0
#     f_min = 0.0
#     filter_length = 1024
#
#     # spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024,
#     #                                                    win_length=win_length, hop_length=hop_length,
#     #                                                    n_mels=80, f_max=8000.0).to(device)
#     resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
#
#     shape = get_dataset_config(args)
#     dataloader, evalloader = get_dataset(args)
#     data = next(iter(dataloader))
#     data, ch_num = audio_path_to_data_hifi(data['wav'], resampler)
#     data = data[:ch_num[0]]
#     spec = mel_spectrogram(data, filter_length, num_mels,
#                           sampling_rate, hop_length,
#                           win_length, f_min, f_max,
#                           center=False)
#
#     plt.figure(figsize=(10, 12))
#     res_mel = spec[0].detach().cpu().numpy()
#     plt.imshow(res_mel, origin='lower')
#     plt.xlabel('time')
#     plt.ylabel('frequency')
#     _ = plt.title('Spectrogram')
#     plt.show()
#     print(spec.max(), spec.min())
#     print(spec.shape)
#     with torch.no_grad():
#         audios = hifigan(spec.to(device)).float()
#         audios = denoiser(audios.squeeze(1), denoising_strength)
#         audios = audios.squeeze(1) #* vocoder_train_setup['max_wav_value']
#         audio_numpy = audios[0].cpu().numpy()
#         write("test_audio/audio.wav", sampling_rate, audio_numpy)