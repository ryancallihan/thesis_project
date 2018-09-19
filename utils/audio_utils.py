import sys
sys.path.append('/mnt/Shared/people/ryan/thesis_project')
import os
import librosa
import logging
import numpy as np
import tensorflow as tf
import scipy.signal
import scipy.io.wavfile
from scipy.fftpack import dct
from scipy.stats import norm
from utils.network_utils import FeatureWriter
import utils.data_processing_utils as dp


def get_signal_wav(wavpath, emphasize=True, pre_emphasis=0.97):
    """
    Gets signal and sample rate from wav file.
    If emphasize is True https://www.quora.com/Why-is-pre-emphasis-i-e-passing-the-speech-signal-through-a-first-order-high-pass-filter-required-in-speech-processing-and-how-does-it-work/answer/Nickolay-Shmyrev?srid=e4nz&share=71ca3e28
    :param wavpath: string
    :param emphasize: boolean
    :param pre_emphasis: float
    :return: (signal, sample_rate)
    """
    sample_rate, signal = scipy.io.wavfile.read(wavpath)  # File assumed to be in the same directory
    if emphasize:
        return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1]), sample_rate
    return signal, sample_rate


def milliseconds2samples(milliseconds, sample_rate):
    """
    Converts list of milliseconds to sample segments
    :param milliseconds: array
    :param sample_rate: int
    :return: array
    """
    samps = (milliseconds / 10000) * sample_rate
    if type(samps) == float:
        return np.int(samps)
    return samps.astype(np.int)


# def samples2milliseconds(samples, sample_rate):
#     """
#     Converts list of milliseconds to sample segments
#     :param milliseconds: array
#     :param sample_rate: int
#     :return: array
#     """
#     return (samples / sample_rate) * 10000


def seconds2samples(seconds, sample_rate):
    """
    Converts list of milliseconds to sample segments
    :param milliseconds: array
    :param sample_rate: int
    :return: array
    """
    return seconds * sample_rate


def samples2seconds(samples, sample_rate):
    """
    Converts list of milliseconds to sample segments
    :param milliseconds: array
    :param sample_rate: int
    :return: array
    """
    return samples / sample_rate


def samples2milliseconds(samples, sample_rate):
    return (samples / np.float32(sample_rate)) * 10000.0


def get_sample_windows(num_samples, sample_rate, len_window):
    """
    Gets sample indices given a time window in seconds
    :param num_samples: int max number of samples in signal
    :param sample_rate: int
    :param len_window: int time in seconds
    :return: numpy array
    """
    len_window = len_window * 10000
    samp_window = milliseconds2samples(len_window, sample_rate)
    windows = list(range(0, num_samples, samp_window))
    if windows[-1] != num_samples:
        windows.append(num_samples)
    return np.array(windows)


def delta(feat, n=2, f=0):
    nl, ml = feat.shape
    gap = n * np.ones(ml)
    gap[0:n] = np.arange(0, n)
    gap[(ml - n):ml] = np.arange((n - 1), -1, -1)
    l_delta_ids = (np.arange(1, ml + 1) + gap).astype(np.int) - 1
    r_delta_ids = (np.arange(1, ml + 1) - gap).astype(np.int) - 1

    gap = f * np.ones(nl)
    gap[0:f] = np.arange(0, f)
    gap[(nl - f):nl] = np.arange((f - 1), -1, -1)
    l_diag_ids = (np.arange(1, nl + 1) - gap).astype(np.int) - 1
    r_diag_ids = (np.arange(1, nl + 1) + gap).astype(np.int) - 1

    l_feat = np.array([[feat[l_diag_ids[i], l_delta_ids[j]] for j in range(ml)] for i in range(nl)])
    r_feat = np.array([[feat[r_diag_ids[i], r_delta_ids[j]] for j in range(ml)] for i in range(nl)])

    return np.subtract(l_feat, r_feat)


def mean_normalize(frames):
    """
    TODO Look at exactly how this is normalizing!!!!
    Normalizes by subtracting the mean from each value.
    :param frames: array
    :return: array
    """
    frames -= (np.mean(frames, axis=0) + 1e-8)
    return frames


def histogram_normalize(feat):
    len_feat = feat.shape[1]
    n_feat = []
    for f in feat:
        feat_idx = np.argsort(f)
        sort_idx = (np.arange(1, len_feat + 1) - 0.5) / len_feat
        sort_val = norm.ppf(sort_idx, 0, 1)
        norm_val = np.zeros((len(feat_idx)))
        for n, idx in enumerate(feat_idx):
            norm_val[idx] = sort_val[n]
        n_feat.append(norm_val)
    return np.array(n_feat)


def dscc(spectrogram, n=2, f=0, normalize=True):
    d = delta(spectrogram, n=n, f=f)
    norm_d = histogram_normalize(d)
    d_cep = dct(norm_d)
    dd_cep = delta(d_cep, n=n, f=f)
    if normalize:
        d_cep, dd_cep = mean_normalize(d_cep), mean_normalize(dd_cep)
    return d_cep, dd_cep


def get_boundary_idx(boundaries):
    """
    Returns a list of tuples with start and end indices of samples
    :param boundaries: list of boundaries
    :return: list
    """
    return list(zip(boundaries[:-1], boundaries[1:]))


def append_features(*args):
    len, height = zip(*[np.array(ar).shape for ar in args[0]])
    # Lens should all be equal
    feats = np.zeros((len[0], sum(height)))
    tmp_h = 0
    for i, f in enumerate(args[0]):
        feats[:, (tmp_h): tmp_h + height[i]] = f
        tmp_h += height[i]
    return feats


def append_features_3d(height, width, *args):
    features = np.zeros((width, height, len(args[0])))
    for c, f in enumerate(args[0]):
        features[:, :, c] = np.float32(f)
    return features


class AudioProcessing:

    def __init__(self,
                 sample_rate=16000):
        self.sample_rate = sample_rate

    def process(self,
                signal,
                audio_features,
                sample_rate=16000,
                feature_concatenation='append',
                pad=True,
                mat_shape=None,
                num_ceps=13,
                spec_log=True,
                nfft=512,
                vad=True):
        num_ceps = num_ceps + 1
        processed_feats = dict()
        if 'spectrogram' in audio_features or 'dspec' in audio_features or 'dscc' in audio_features:
            if not spec_log:
                processed_feats['spectrogram'] = np.abs(
                    librosa.core.stft(signal, n_fft=nfft, hop_length=160, win_length=400))
            else:
                processed_feats['logspec'] = np.log(
                    np.abs(librosa.core.stft(signal, n_fft=nfft, hop_length=160, win_length=400)))

        if 'melspec' in audio_features:
            processed_feats['melspec'] = librosa.feature.melspectrogram(signal, sample_rate, n_fft=nfft,
                                                                        hop_length=160, n_mels=40, fmin=133, fmax=6955)
        elif 'logmel' in audio_features:
            processed_feats['logmel'] = np.log(
                librosa.feature.melspectrogram(signal, sample_rate, n_fft=nfft, hop_length=160, n_mels=40, fmin=133,
                                               fmax=6955))

        if 'mfcc' in audio_features or 'delta' in audio_features or 'deltadelta' in audio_features:
            processed_feats['mfcc'] = librosa.feature.mfcc(signal, self.sample_rate, n_fft=nfft, hop_length=160,
                                                           n_mfcc=40, fmin=133, fmax=6955)[1:num_ceps]
        if 'delta' in audio_features or 'deltadelta' in audio_features:
            processed_feats['delta'] = delta(processed_feats['mfcc'], n=2, f=0)
            if 'deltadelta' in audio_features:
                processed_feats['deltadelta'] = delta(processed_feats['delta'], n=2, f=0)

        if 'dspec' in audio_features or 'dscc' in audio_features:
            d, dd = dscc(processed_feats['spectrogram'], n=2, f=0)
            processed_feats['dspec'] = d[1:num_ceps]
            processed_feats['dscc'] = dd[1:num_ceps]

        for f in audio_features:
            if len(processed_feats[f]) < 0:
                print('%s is less than 0.' % f)
                return None

            # Simple VAD based on the energy
            if vad:
                E = librosa.feature.rmse(signal, frame_length=nfft, hop_length=160)
                threshold = np.mean(E) / 2 * 1.04
                vad_segments = np.nonzero(E > threshold)
                if vad_segments[1].size != 0:
                    processed_feats[f] = processed_feats[f][:, vad_segments[1]]

            if pad:
                processed_feats[f] = dp.pad_matrix(processed_feats[f], mat_shape)

            # processed_feats[f] = np.array(processed_feats[f])

        if len(audio_features) == 1:
            features = processed_feats[audio_features[0]].T

        elif feature_concatenation == '3d':
            heights = [processed_feats[af].shape[0] for af in audio_features]
            max_height = np.max(heights)
            for af in audio_features:
                if processed_feats[af].shape[0] != max_height:
                    processed_feats[af] = dp.pad_matrix(processed_feats[af], (max_height, processed_feats[af].shape[1]))

            height = max_height
            width = processed_feats[audio_features[0]].shape[1]
            features = append_features_3d(height, width, [processed_feats[f].T for f in audio_features])

        else:
            features = np.array(
                append_features([processed_feats[k].T for k in audio_features])
            )
        # features = processed_feats[audio_features[0]].T
        return features

    def write_feat_labels(self, data_dir, save_path, feature_type, speed, volume):

        code2idx = {
            'EGY': 3,
            'GLF': 1,
            'LAV': 0,
            'MSA': 4,
            'NOR': 2
        }
        wav_ending = '_s%s_v%s.wav' % (str(speed), str(volume))
        print('WAV ENDING', wav_ending)
        writer = FeatureWriter(save_path, initialize=True)
        for dir in os.listdir(data_dir):
            logging.info('Working on %s' % dir, code2idx[dir])
            # if not os.path.isdir(os.path.join(data_dir, dir)):
            #     break
            for i, file in enumerate(os.listdir(os.path.join(data_dir, dir))):
                if i % 1000 == 0:
                    logging.info('Working on %d of %d' % (i, len(os.listdir(os.path.join(data_dir, dir)))))
                if wav_ending in file:

                    signal, sr = librosa.core.load(os.path.join(data_dir, dir, file), sr=16000, mono=True, dtype='float')
                    features = self.process(signal=signal,
                                            audio_features=[feature_type],
                                            sample_rate=sr,
                                            pad=False,
                                            num_ceps=13,
                                            spec_log=False,
                                            vad=True)
                    writer.write_feature(file, features, code2idx[dir])
        writer.close()
