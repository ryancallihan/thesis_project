import sys

sys.path.append('/mnt/Shared/people/ryan/thesis_project')
import os
import librosa
import numpy as np
import logging
from random import shuffle
from rnn.config import Config
from rnn.train import train
from utils.audio_utils import AudioProcessing
from utils.data_processing_utils import split_matrices
from utils.network_utils import logdir


if __name__ == '__main__':
    """
    Trains RNN on MGB3 data
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='rnn.log',
                        filemode='w')

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


    # train_path = 'test.txt'
    # train_path = 'mgb_train_logmel_s1.0_v1.0.tsv'
    # test_path = 'mgb_dev_mfcc_s1.0_v1.0.tsv'
    # test_x, test_y, _ = FeatureReader(test_path).read()
    #
    # SPEED_LIST = [0.9, 1.0, 1.1]
    # VOL_LIST = [0.125, 1.0, 2.0]
    # train_x = []
    # train_y = []
    # # test_x = []
    # # test_y = []
    # for s in SPEED_LIST:
    #   for v in VOL_LIST:
    #       print('Loading in speed: %s and volume: %s' % (str(s), str(v)))
    #       x, y, _ = FeatureReader('mgb_train_mfcc_s%s_v%s.tsv' % (str(s), str(v))).read()
    #       train_x.extend(x)
    #       train_y.extend(y)
    #       # if s == 1.0 and v == 1.0:
    #       #   test_x.extend(x[int(len(x)*.9):])
    #       #   test_y.extend(y[int(len(x)*.9):])
    #       logging.info('train shape: %s | test shape: %s' % (str(len(train_x)), str(len(test_x))))
    #
    # from sklearn.model_selection import train_test_split
    #
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, shuffle=Config.shuffle, test_size=Config.test_size)

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
                    features = ap.process(signal=signal,audio_features=config.audio_features,feature_concatenation='append',sample_rate=sr,pad=False,num_ceps=13,spec_log=False,vad=True)
                    x.append(features)
                    y.append(code2idx[dir])
            train_idx = list(range(int(len(y) * (1.0 - Config.test_size))))
            test_idx = list(range(len(train_idx), len(y)))
            tr_x, tr_y = zip(*[(x[i], y[i]) for i in train_idx])
            te_x, te_y = zip(*[(x[i], y[i]) for i in test_idx])
            train_x.extend(tr_x)
            train_y.extend(tr_y)
            test_x.extend(te_x)
            test_y.extend(te_y)
        return train_x, test_x, train_y, test_y


    audio_feat_schedule = [['logmel'], ['mfcc']]
    # audio_feat_schedule = [['mfcc', 'dscc'], ['logmel', 'dscc']]
    for af in audio_feat_schedule:
        Config.audio_features = af
        # Config.audio_features = audio_feat_schedule[1]

        train_dir = '/mnt/Shared/people/ryan/varDial2017_adi/train'
        # train_dir = 'C:/Users/ryanc/Documents/corpora/mgb_subset/train'
        train_x, test_x, train_y, test_y = process_files(train_dir, config=Config, speed=1.0, volume=1.0)
        # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, shuffle=Config.shuffle, test_size=Config.test_size)
        train_x, train_y, train_shapes = split_matrices(train_x, train_y, 512, input='append')
        test_x, test_y, test_shapes = split_matrices(test_x, test_y, 512, input='append')
        logging.info('Retrieved filenames and labels')
        if Config.shuffle:
            train_idx = list(range(len(train_y)))
            test_idx = list(range(len(test_y)))
            shuffle(train_idx)
            shuffle(test_idx)
            train_x = np.array([train_x[i] for i in train_idx])
            train_y = np.array([train_y[i] for i in train_idx])
            test_x = np.array([test_x[i] for i in test_idx])
            test_y = np.array([test_y[i] for i in test_idx])
        model_dir = logdir('arabic_5sec',Config.architecture_type,'-'.join(Config.audio_features),main_dir='models')
        train_x = train_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        num_labels = len(set(train_y))
        logging.info('TRAIN FEATURE SHAPE: %s' % str(train_x.shape))
        logging.info('TEST FEATURE SHAPE: %s' % str(test_x.shape))
        logging.info('TRAIN LABEL SHAPE: %s' % str(train_y.shape))
        logging.info('TEST LABEL SHAPE: %s' % str(test_y.shape))
        logging.info('NUM LABELS: %s' % str(num_labels))
        logging.info('TRAIN LABELS: %s' % str(set(train_y)))
        logging.info('TEST LABELS: %s' % str(set(test_y)))

        train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, config=Config, num_classes=num_labels,model_dir=model_dir)
