import sys
sys.path.append('/mnt/Shared/people/ryan/thesis_project')
import numpy as np
import logging
from random import shuffle
from cnn_1d.config import Config
from cnn_1d.train import train
from utils.corpora.mgb_utils import process_files
from utils.network_utils import logdir
from utils.data_processing_utils import split_matrices


if __name__ == '__main__':
    """
    Trains CNN1D on MGB3 data
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='cnn1d_cnn.log',
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
    #
    # # audio_concat = ['3d', 'append']
    # audio_feat_schedule = [['logmel'], ['mfcc'], ['logmel', 'dscc'], ['mfcc', 'dscc']]
    audio_feat_schedule = [['logmel', 'dscc'], ['mfcc', 'dscc']]
    audio_concat = ['append']
    # for ac in audio_concat:
    #     Config.feature_concatenation = ac
    for af in audio_feat_schedule:
        Config.audio_features = af
        # Config.audio_features = audio_feat_schedule[1]

        # Config.audio_features = ['logmel']
        Config.feature_concatenation = 'append'
        train_dir = '/mnt/Shared/people/ryan/varDial2017_adi/train'
        # train_dir = '/mnt/Shared/people/ryan/mgb_subset/train'
        train_x, test_x, train_y, test_y = process_files(train_dir, config=Config, speed=1.0, volume=1.0)
        print('train x ', np.array(train_x).shape)
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

        Config.gru = False
        Config.architecture_type = 'cnn_1d'
        model_dir = logdir('arabic_5sec', Config.architecture_type, Config.feature_concatenation,
                           '-'.join(Config.audio_features), main_dir='models')

        train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, config=Config, num_classes=num_labels,model_dir=model_dir)

