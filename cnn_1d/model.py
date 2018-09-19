import tensorflow as tf
import logging
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
        self.mode = None
        self.config = config
        self.num_classes = num_classes
        self.logdir = logdir

    def model_fn(self, features, labels, mode):
        """
        Builds the neural net
        :param features: dict
        :param labels: list
        :param mode: object
        :return:
        """

        features = features['features']

        # if len(features.shape) < 4:
        #     # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        #     features = tf.reshape(features, shape=[-1, int(features.shape[1]), int(features.shape[2]), 1])
        self.mode = mode
        logits = self.conv_net(features)

        predicted_indices = tf.argmax(input=logits, axis=1)

        # TRAINING FUNCTIONS
        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):

            # Define loss and optimizer
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.cast(labels, dtype=tf.int32), name='loss'))

            # L2 Regularization
            if self.config.l2_regularization:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                loss = tf.reduce_mean(loss + self.config.l2_beta * l2_loss, name='l2Reg')
            metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_indices, name='acc_op')}
            tf.summary.scalar('accuracy', metrics['accuracy'][1])
            tf.summary.scalar('loss', loss)
            tf.summary.merge_all()
            tf.summary.FileWriter(self.logdir)

        if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
            probabilities = tf.nn.softmax(logits, name='softmax_tensor')

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_indices,
                    'probabilities': probabilities
                }
                export_outputs = {
                    'predictions': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(
                    mode, predictions=predictions, export_outputs=export_outputs)

        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = self.config.learning_rate
            if self.config.learning_rate_decay:
                global_step = tf.Variable(0, trainable=False)
                lr = tf.train.exponential_decay(learning_rate=lr,
                                                global_step=global_step,
                                                decay_steps=self.config.learning_rate_decay,
                                                decay_rate=self.config.lr_decay_rate,
                                                staircase=True,
                                                name='learningRateDecay')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam')
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            # Evaluate the accuracy of the model
            # metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_indices, name='acc_op')}
            # tf.summary.scalar('accuracy', metrics['accuracy'][1])
            # tf.summary.merge_all()
            # tf.summary.FileWriter(self.logdir)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics)

    def conv_layer_1d(self,
                      input_data,
                      filters,
                      kernel,
                      stride_conv,
                      size_pool,
                      stride_pool,
                      count,
                      pad_conv='same',
                      pad_pool='valid'):
        """
        Creates a convolutional layer.
        :param input_data:
        :param filters:
        :param kernel:
        :param stride_conv:
        :param size_pool:
        :param stride_pool:
        :param count:
        :param pad_conv:
        :param pad_pool:
        :return:
        """
        conv_layer = tf.layers.conv1d(inputs=input_data,
                                      filters=filters,
                                      kernel_size=kernel,
                                      strides=stride_conv,
                                      padding=pad_conv,
                                      activation=None,
                                      use_bias=True,
                                      name='conv1d' + str(count))
        if self.config.batch_norm:
            conv_layer = tf.layers.batch_normalization(conv_layer, axis=-1,
                                                       training=self.mode == tf.estimator.ModeKeys.TRAIN,
                                                       name='ConvBatchNorm' + str(count))
            logging.info(conv_layer)
        logging.info(conv_layer)
        conv_layer = tf.nn.relu(conv_layer, name='ReLU' + str(count))
        logging.info(conv_layer)
        if size_pool is not None:
            conv_layer = tf.layers.max_pooling1d(inputs=conv_layer,
                                                 pool_size=size_pool,
                                                 strides=stride_pool,
                                                 padding=pad_pool,
                                                 name='maxPool' + str(count))
            logging.info(conv_layer)
        # if self.config.batch_norm:
        #     conv_layer = tf.layers.batch_normalization(conv_layer, axis=-1,
        #                                                training=self.mode == tf.estimator.ModeKeys.TRAIN,
        #                                                name='ConvBatchNorm' + str(count))
        #     logging.info(conv_layer)
        return conv_layer

    def get_conv_layers(self, input_data):
        """
        Creates multi conv layers
        :param input_data:
        :return:
        """
        count = 1
        with tf.variable_scope('ConvLayers'):
            for f, k, sc, sp, stp, pc, pp in zip(*(
                    self.config.num_filters,
                    self.config.kernel_sizes,
                    self.config.strides,
                    self.config.pool_sizes,
                    self.config.pool_strides,
                    self.config.padding,
                    self.config.pool_padding)
                                                 ):
                with tf.variable_scope('ConvLayer%d' % count):
                    input_data = self.conv_layer_1d(input_data, f, k, sc, sp, stp, count, pc, pp)
                    count += 1
            return input_data

    def create_gru_layers(self, inputs, activation=tf.nn.relu):
        """
        Dynamically creates GRU layers, bidirectional and otherwise.
        :param activation: activation object
        :param inputs: list
        :return: hidden layers
        """
        with tf.variable_scope('GRULayers'):
            cells = [rnn.GRUCell(size, activation=activation) for size in
                     self.config.fully_connected_sizes]  # Configures hidden cells.
            cells = [tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=self.config.dropout) for c in cells]
            # if self.config.batch_norm:
            #     cells = [tf.layers.batch_normalization(c, axis=-1, training=self.mode == tf.estimator.ModeKeys.TRAIN,
            #                                            name='GRUBatchNorm' + str(i + 1)) for i, c in enumerate(cells)]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)  # Combines cells
            _, states = tf.nn.dynamic_rnn(cell=cells, inputs=inputs, dtype=tf.float32)
            return states[-1]

    def conv_net(self, features):
        """
        Creates the layers for the CNN
        :param features: list
        :return: tensors
        """
        # Define a scope for reusing the variables
        with tf.variable_scope(self.config.network_name):
            # This is of variable size according to the Configs
            # conv_layers = self.get_conv_layers(features)
            conv_layers = self.get_conv_layers(features)

            # TODO Test this out. Might need to add size to Config. Uncomment to use
            if self.config.gru:
                outputs = self.create_gru_layers(inputs=conv_layers)
                logging.info(outputs)
            else:
                outputs = tf.contrib.layers.flatten(conv_layers)
                logging.info(outputs)

                # Fully connected layer (in tf contrib folder for now)
                # TODO  should I make the fc layer variable?
                with tf.variable_scope('FullyConnectedLayers'):
                    for i, s in enumerate(self.config.fully_connected_sizes):
                        with tf.variable_scope('FullyConnectedLayer%d' % (i + 1)):
                            outputs = tf.layers.dense(outputs, s, activation=None, name='FCLayer' + str(i + 1))
                            logging.info(outputs)
                            if self.config.batch_norm:
                                outputs = tf.layers.batch_normalization(outputs, axis=-1,
                                                                        training=self.mode == tf.estimator.ModeKeys.TRAIN,
                                                                        name='FCBatchNorm' + str(i + 1))
                            outputs = tf.nn.relu(outputs, name='FCReLU' + str(i + 1))
                            logging.info(outputs)
                            logging.info(outputs)
                            outputs = tf.layers.dropout(outputs, rate=self.config.dropout,
                                                        training=self.mode == tf.estimator.ModeKeys.TRAIN)
                            logging.info(outputs)

            # Output layer, class prediction
            outputs = tf.layers.dense(outputs, self.num_classes, activation=None, name='softmax')
            logging.info(outputs)
            return outputs
