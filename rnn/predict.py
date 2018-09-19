import tensorflow as tf
import numpy as np
from rnn.model import Model


class Predict:

    def __init__(self, model_dir, config, num_classes):
        """

        :param model_dir: string
        :param config: Config object
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
    Returns class ids from prediction iterator
    :param predictions: iterator
    :return: list
    """
    return np.array([p['class_ids'][0] for p in predictions])


def get_logits(predictions):
    """
    Returns logits from prediction iterator
    :param predictions: iterator
    :return: list
    """
    return np.array([p['logits'][0] for p in predictions])
