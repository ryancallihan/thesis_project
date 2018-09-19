import os
import logging
import shutil
import numpy as np


def logdir(*args, main_dir=None):
    """
    Creates a logging path which includes necessary hyperparameter information
    :param config: config object
    :return: string
    """
    model_name = '_'.join([str(a) for a in args])
    if main_dir is not None:
        model_name = os.path.join(main_dir, model_name)
    return model_name


def copy_configs(src_file, dst_dir):
    shutil.copy(src_file, dst_dir)


class FeatureReader:

    def __init__(self, filename):
        self.filename = filename
        self.file = self.open_file()

    def open_file(self):
        return open(self.filename, 'r', encoding='utf-8')

    def close(self):
        self.file.close()

    def read(self):
        features = []
        labels = []
        information = []
        lines = self.file.readlines()
        logging.info('%d items in %s' % (len(lines), self.filename))
        for i,line in enumerate(lines):
            if i % 1000 == 0:
                logging.info('Working on %d of %d' % (i, len(lines)))
            line = line.strip()
            if line is None:
                continue
            feat_tmp = []
            info, f, lab = line.split('\t')
            lines_to_conv = f.split('|')
            for l in lines_to_conv:
                feat_tmp.append([np.float32(e) for e in l.split(' ')])
            features.append(np.array(feat_tmp))
            labels.append(np.int32(lab))
            information.append(info)
        self.close()
        return np.array(features), np.array(labels), information


class FeatureWriter:

    def __init__(self, filename, initialize=False):
        self.filename = filename
        if initialize:
            self.file = self.initialize_file()

    def initialize_file(self):
        return open(self.filename, 'w', encoding='utf-8')

    def write_feature(self, info, feature, label):
        line_to_write = '%s\t' % info
        for i, line in enumerate(feature):
            line_to_write += ' '.join([str(l) for l in line])
            if i+1 != len(feature):
                line_to_write += '|'
        self.file.write('%s\t%s\n' % (line_to_write, label))

    def close(self):
        self.file.close()

