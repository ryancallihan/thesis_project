import sys
sys.path.append('/mnt/Shared/people/ryan/thesis_project')
import os
import librosa
import numpy as np
import logging
from random import shuffle
from cnn_1d.config import Config as Config_cnn1d
from cnn_1d.train import train as train_cnn1d
from rnn.config import Config as Config_rnn
from rnn.train import train as train_rnn
from utils.audio_utils import AudioProcessing
from utils.network_utils import logdir
from utils.data_processing_utils import split_matrices

if __name__ == '__main__':
    """
    Trains CNN, CNN+RNN, RNN on FRED-S data
    """

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='freds.log',
                        filemode='w+')

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


    def process_files(data_dir, config):
        code2idx = {
            'MID': 0,
            'N': 3,
            'SCL': 2,
            'SE': 1,
            'SW': 4
        }
        X = []
        Y = []

        # wav_ending = '_s%s_v%s.wav' % (str(speed), str(volume))
        for dir in os.listdir(data_dir):
            logging.info('Working on %s' % str(config.audio_features))
            x = []
            y = []
            logging.info('Working on %s' % dir)
            # if not os.path.isdir(os.path.join(data_dir, dir)):
            #   break
            ap = AudioProcessing()
            for i, file in enumerate(os.listdir(os.path.join(data_dir, dir))):
                if i % 5000 == 0:
                    logging.info('Working on %d of %d' % (i, len(os.listdir(os.path.join(data_dir, dir)))))
                if '.wav' in file:
                    try:
                        signal, sr = librosa.core.load(os.path.join(data_dir, dir, file), sr=16000, mono=True, dtype='float')
                        features = ap.process(signal=signal,audio_features=config.audio_features,feature_concatenation='append',sample_rate=sr,pad=False,num_ceps=13,spec_log=False,vad=True)
                        x.append(features)
                        y.append(code2idx[dir])
                    except ValueError:
                        logging.error('Value Error :(')
                    except EOFError:
                        logging.error('EOF Error :(')
            X.extend(x)
            Y.extend(y)
        return X, Y


    audio_feat_schedule = [['logmel'], ['mfcc'], ['mfcc', 'dscc'], ['logmel', 'dscc']]
    # audio_concat = ['3d', 'append']
    # audio_feat_schedule = [['mfcc', 'dscc'], ['logmel', 'dscc']]
    audio_concat = ['append']
    for ac in audio_concat:
        # Config_rnn.feature_concatenation = ac
        # Config_rnn.feature_concatenation = ac
        for af in audio_feat_schedule:
            Config_rnn.audio_features = af
            # Config_rnn.audio_features = af
        # Config.audio_features = audio_feat_schedule[1]

            train_dir = '/mnt/Shared/people/ryan/fred-s/train'
            test_dir = '/mnt/Shared/people/ryan/fred-s/test'

            train_x, train_y = process_files(train_dir, config=Config_rnn)
            test_x, test_y = process_files(test_dir, config=Config_rnn)
            print('train x ', np.array(train_x).shape)

            train_x, train_y, train_shapes = split_matrices(train_x, train_y, 512, input='append')
            test_x, test_y, test_shapes = split_matrices(test_x, test_y, 512, input='append')
            logging.info('Retrieved filenames and labels')
            if Config_rnn.shuffle:
                train_idx = list(range(len(train_y)))
                test_idx = list(range(len(test_y)))
                shuffle(train_idx)
                shuffle(test_idx)
                train_x = np.array([train_x[i] for i in train_idx])
                train_y = np.array([train_y[i] for i in train_idx])
                test_x = np.array([test_x[i] for i in test_idx])
                test_y = np.array([test_y[i] for i in test_idx])

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

            # model_dir = logdir('english',Config_rnn.architecture_type,Config_rnn.feature_concatenation,'-'.join(Config_rnn.audio_features),main_dir='fred-models')
            # train_rnn(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, config=Config_rnn, num_classes=num_labels,
            #       model_dir=model_dir)

            Config_cnn1d.dropout = 0.5
            Config_cnn1d.num_filters = [128, 256, 512]
            Config_cnn1d.kernel_sizes = [5, 7, 2]
            Config_cnn1d.strides = [1, 2, 1]
            Config_cnn1d.padding = ['valid', 'valid', 'valid']
            Config_cnn1d.pool_sizes = [2, 2, 2]
            # pool_sizes = [None, None, None]
            Config_cnn1d.pool_strides = [2, 2, 2]
            Config_cnn1d.pool_padding = ['valid', 'valid', 'valid']
            Config_cnn1d.fully_connected_sizes = [128, 64]

            Config_cnn1d.gru = False
            Config_cnn1d.architecture_type = 'cnn_1d_5sec'
            model_dir = logdir('english_original',Config_cnn1d.architecture_type,Config_cnn1d.feature_concatenation,'-'.join(Config_cnn1d.audio_features),main_dir='fred-models')
            train_cnn1d(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, config=Config_cnn1d, num_classes=num_labels,model_dir=model_dir)

            Config_cnn1d.gru = True
            Config_cnn1d.architecture_type = 'cnn-rnn_1d_5sec'
            model_dir = logdir('english_original', Config_cnn1d.architecture_type, Config_cnn1d.feature_concatenation,'-'.join(Config_cnn1d.audio_features), main_dir='fred-models')
            train_cnn1d(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, config=Config_cnn1d, num_classes=num_labels,model_dir=model_dir)

            Config_rnn.architecture_type = 'rnn_5sec'
            model_dir = logdir('english_original', Config_rnn.architecture_type,'-'.join(Config_rnn.audio_features), main_dir='fred-models')
            train_rnn(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, config=Config_rnn, num_classes=num_labels,model_dir=model_dir)

