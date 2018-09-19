import tensorflow as tf
from tensorflow.contrib import rnn


class Model:

    def __init__(self,
                 config,
                 num_classes,
                 logdir):
        """
        :param config: Configs from config.py
        :param num_classes: int
        :param logdir: string
        """
        self.config = config
        self.num_classes = num_classes
        self.logdir = logdir
        self.is_training = False

    def model_fn(self, features, labels, mode):
        """
        Builds the neural net
        :param features: dict
        :param labels: list
        :param mode: object
        :return: estimator object
        """
        if mode == tf.estimator.ModeKeys.TRAIN:  # If training, enables backprop, dropout, etc.
            self.is_training = True

        # features = tf.reshape(features['features'], shape=[-1, int(features['features'].shape[1]), 1])
        print('features', features)

        logits = self.rnn_net(features['features'], reuse=False)  # Loads model

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)  # TODO should these be reversed? Do we use the logits to train????
        pred_probs = tf.nn.softmax(logits)

        predictions = {
            'class_ids': pred_classes[:, tf.newaxis],
            'probabilities': pred_probs,
            'logits': logits,
        }

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:  # Returns predictions from model
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # TRAINING FUNCTIONS

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32), name='loss'))

        # L2 Regularization
        if self.config.l2_regularization:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss = tf.reduce_mean(loss + self.config.l2_beta * l2_loss, name='l2Reg')

        lr = self.config.learning_rate
        if self.config.learning_rate_decay:
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(learning_rate=lr,
                                            global_step=global_step,
                                            decay_steps=self.config.lr_decay_steps,
                                            decay_rate=self.config.lr_decay_rate,
                                            staircase=True,
                                            name='learningRateDecay')

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='adamOptimizer')
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # Evaluate the accuracy of the model TODO improve this.
        accuracy = tf.metrics.accuracy(labels=labels, predictions=pred_classes, name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('loss', loss)

        tf.summary.merge_all()
        tf.summary.FileWriter(self.logdir)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        return estim_specs

    def hidden_layers(self, inputs, activation=tf.nn.relu):
        """
        Dynamically creates GRU layers, bidirectional and otherwise.
        :param activation: activation object
        :param inputs: list
        :return: hidden layers
        """
        # BIDIRECTIONAL GRU
        if self.config.cell_type == 'gru':
            rnn_cell = rnn.GRUCell
        elif self.config.cell_type == 'lstm':
            rnn_cell = rnn.LSTMCell
        elif self.config.cell_type == 'lstm_norm':
            rnn_cell = rnn.LayerNormBasicLSTMCell
        else:
            rnn_cell = rnn.RNNCell

        cells = [rnn_cell(size, activation=activation) for size in self.config.hidden_sizes]  # Configures hidden cells.
        if self.config.batch_norm:
            cells = [tf.layers.batch_normalization(c, axis=-1, training=self.is_training,
                                                   name='BN' + str(i)) for i, c in enumerate(cells)]
        cells = [tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=self.config.dropout) for c in cells]
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)  # Combines cells
        return tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    def rnn_net(self, features, reuse):
        """
        Creates the layers for the RNN
        :param features: list
        :param reuse: boolean
        :return: tensors
        """
        # Define a scope for reusing the variables
        with tf.variable_scope('%sNet' % self.config.cell_type, reuse=reuse):

            # TF Estimator input is a dict, in case of multiple inputs
            _, states = self.hidden_layers(features)

            output = tf.layers.dense(states[-1], self.num_classes, activation=None)

            return output
