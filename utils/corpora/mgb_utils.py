import os
import logging
import librosa
from utils.audio_utils import AudioProcessing
from utils.audio_utils import get_signal_wav


class MGBFeature:

    def __init__(self, wav_path=None, dialect_code=None, dialect=None, dialect_label=None, get_signal=False):
        self._wav_path = wav_path
        self._dialect_code = dialect_code
        self._dialect = dialect
        self._dialect_label = dialect_label
        self._signal, self._sample_rate = get_signal_wav(self.wav_path, emphasize=False)
        # if get_signal:
        #     sig, sr = get_signal_wav(self.wav_path, emphasize=False)
        #     self.signal(sig)
        #     self.sample_rate(sr)
        if self.dialect_code is not None or self.dialect is not None or self.label is not None:
            self.generate_dialect_info()

    def generate_dialect_info(self):
        if self.label is not None:
            self._dialect_code = DialectMapping.idx2code[self.label]
            self._dialect = DialectMapping.code2dialect[self.dialect_code]
        elif self.dialect_code is not None:
            self._dialect_label = DialectMapping.code2idx[self.dialect_code]
            self._dialect = DialectMapping.code2dialect[self.dialect_code]
        elif self.dialect is not None:
            self._dialect_code = DialectMapping.dialect2code[self.dialect]
            self._dialect_label = DialectMapping.code2idx[self.dialect_code]

    @property
    def wav_path(self):
        return self._wav_path

    @wav_path.setter
    def wav_path(self, wav_path):
        self._wav_path = wav_path

    @property
    def label(self):
        return self._dialect_label

    @property
    def dialect_code(self):
        return self._dialect_code

    @property
    def dialect(self):
        return self._dialect

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        self._signal = signal

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = sample_rate


def get_files_labels(datadir, train=True):
    mgb_objects = []

    if train:
        datadir = os.path.join(datadir, 'train')
    else:
        datadir = os.path.join(datadir, 'dev')
    for dialect in os.listdir(datadir):
        print('Getting %s files' % dialect)
        if not os.path.isdir(os.path.join(datadir, dialect)):
            continue
        dialect_files = os.listdir(os.path.join(datadir, dialect))
        for i, f in enumerate(dialect_files):
            if i % 500 == 0:
                print('%d of %d for %s' % (i, len(dialect_files), dialect))
            if not f.endswith('.wav'):
                continue

            mgb_objects.append(MGBFeature(wav_path=os.path.join(datadir, dialect, f),
                                          dialect_code=dialect,
                                          get_signal=True))
    return mgb_objects


class DialectMapping:
    dialect2code = {
        'Egyptian': 'EGY',
        'Gulf': 'GLF',
        'Levantine': 'LAV',
        'North-African': 'NOR',
        'Modern Standard Arabic': 'MSA'
    }

    code2dialect = {
        'EGY': 'Egyptian',
        'GLF': 'Gulf',
        'LAV': 'Levantine',
        'MSA': 'Modern Standard Arabic',
        'NOR': 'North-African'
    }

    code2idx = {
        'EGY': 3,
        'GLF': 1,
        'LAV': 0,
        'MSA': 4,
        'NOR': 2
    }

    idx2code = {
        0: 'LAV',
        1: 'GLF',
        2: 'NOR',
        3: 'EGY',
        4: 'MSA'
    }


def process_files(data_dir, config, speed, volume):
    code2idx = {'EGY': 3, 'GLF': 1, 'LAV': 0, 'MSA': 4, 'NOR': 2}
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    wav_ending = '_s%s_v%s.wav' % (str(speed), str(volume))
    for dir in os.listdir(data_dir):
        x = []
        y = []
        logging.info('Working on %s' % dir)
        # if not os.path.isdir(os.path.join(data_dir, dir)):
        #   break
        ap = AudioProcessing()
        for i, file in enumerate(os.listdir(os.path.join(data_dir, dir))):
            if i % 1000 == 0:
                logging.info('Working on %d of %d' % (i, len(os.listdir(os.path.join(data_dir, dir)))))
            if wav_ending in file:
                signal, sr = librosa.core.load(os.path.join(data_dir, dir, file), sr=16000, mono=True, dtype='float')
                features = ap.process(signal=signal,audio_features=config.audio_features,feature_concatenation=config.feature_concatenation,sample_rate=sr,pad=False,num_ceps=13,spec_log=False,vad=True)
                x.append(features)
                y.append(code2idx[dir])

        train_idx = list(range(int(len(y) * (1.0 - config.test_size))))
        test_idx = list(range(len(train_idx), len(y)))
        tr_x, tr_y = zip(*[(x[i], y[i]) for i in train_idx])
        te_x, te_y = zip(*[(x[i], y[i]) for i in test_idx])
        train_x.extend(tr_x)
        train_y.extend(tr_y)
        test_x.extend(te_x)
        test_y.extend(te_y)
    return train_x, test_x, train_y, test_y

if __name__ == '__main__':
    datadir = 'C:/Users/ryanc/Documents/corpora/varDial2017_adi'

    m = get_files_labels(datadir)
