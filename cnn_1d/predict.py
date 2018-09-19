import tensorflow as tf
import numpy as np
from itertools import groupby
from cnn_1d.model import Model


class Predict:

    def __init__(self, model_dir, config, num_classes):
        """
        :param model_dir: String
        :param config: Config object
        :param num_classes: int
        """
        self.config = config
        self.model = Model(config, num_classes, model_dir)
        self.estimator = tf.estimator.Estimator(model_fn=self.model.model_fn, model_dir=model_dir)

    def predict(self, features):
        """
        Predicts labels, logits, and probabilties.
        :param features: list
        :return: An iterator with predictions
        """

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'features': features}, shuffle=False)
        return self.estimator.predict(input_fn)


def get_class_ids(predictions):
    """
    Gets class ids from predictions. Filters out predictions for samples with more than one entry
    :param predictions:
    :param sample_times:
    :return:
    """
    return [p['class_ids'] for p in predictions]


def get_logits(predictions, sample_times):
    """
    Gets logits from predictions. Filters out predictions for samples with more than one entry
    :param predictions: iterator
    :param sample_times: list
    :return: (list, list)
    """
    logits = [p['logits'] for p in predictions]
    sample_times_grouped = [list(g) for i, g in groupby(sample_times[:-1])]
    new_class_ids = []
    idx = 0
    for st in sample_times_grouped:  # Last time is punct.
        if len(st) != 1:
            idx += len(st) - 1
        new_class_ids.append(logits[idx])
        idx += 1
    return np.array(new_class_ids), [i[0] for i in sample_times_grouped]
