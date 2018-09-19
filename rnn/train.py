import logging
import tensorflow as tf
from rnn.model import Model
# import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)


def train(train_x, train_y, test_x, test_y, config, num_classes, model_dir):
    """

    :param train_x: matrix
    :param train_y: list
    :param test_x: matrix
    :param test_y: list
    :param config: Config object
    :param num_classes: int
    :param model_dir: str
    :return:
    """
    ####### LOAD AND PROCESS DATA #########
    logging.info('Beginning training')
    # y_actual = pd.Series(test_y, name='Actual')
    num_steps = int(train_x.shape[0] / config.batch_size)  # Figure out how many steps in epoch
    logging.info('NUM STEPS: %d' % num_steps)

    config.lr_decay_steps = num_steps

    with tf.device('/cpu:0'):
        input_fn_train = tf.estimator.inputs.numpy_input_fn(
            x={'features': train_x},
            y=train_y,
            batch_size=config.batch_size,
            num_epochs=None,
            shuffle=config.shuffle)

        input_fn_test = tf.estimator.inputs.numpy_input_fn(
            x={'features': test_x
               },
            y=test_y,
            shuffle=False)
    with tf.device('/cpu:1'):
        # Declare model and pass in number of classes for model architecture.
        model = Model(config, num_classes, model_dir)

        session_config = tf.ConfigProto(intra_op_parallelism_threads=config.intra_op_parallelism_threads,
                                        inter_op_parallelism_threads=config.inter_op_parallelism_threads,
                                        allow_soft_placement=True,
                                        device_count={'CPU': config.device_count_cpu}
                                        )

        estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                           model_dir=model_dir,
                                           config=tf.estimator.RunConfig().replace(
                                               save_summary_steps=config.save_summary_steps,
                                               session_config=session_config)
                                           )
    '''
    TRAIN MODEL
    '''
    # Train the Model

    for epoch in range(config.epochs):
        # for substep in range(int(num_steps/2000)):
            estimator.train(input_fn_train, steps=num_steps)  # Train for one epoch

            '''
            EVALUATE ON TEST DATA
            '''
            e = estimator.evaluate(input_fn_test)  # Eval after epoch
            # preds = pd.Series([p['class_ids'][0] for p in estimator.predict(input_fn_test)], name='Predicted')
            # df_confusion = pd.crosstab(y_actual, preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
            # print(df_confusion)

            logging.info("Testing Accuracy: %.2f | Testing Loss: %.2f" % (e['accuracy'], e['loss']))
