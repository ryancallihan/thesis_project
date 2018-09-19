import logging
import tensorflow as tf
from cnn_1d.model import Model
from sklearn.metrics import confusion_matrix, accuracy_score


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

    num_steps = int(train_x.shape[0] / config.batch_size)  # Figure out how many steps in epoch
    logging.info('NUM STEPS: %d' % num_steps)

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

    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={'features': train_x},
        y=train_y,
        batch_size=config.batch_size,
        num_epochs=None,
        shuffle=config.shuffle)

    input_fn_test = tf.estimator.inputs.numpy_input_fn(
        x={'features': test_x},
        y=test_y,
        batch_size=config.batch_size,
        num_epochs=1,
        shuffle=False)

    '''
    TRAIN MODEL
    '''
    # Train the Model

    # with open('cnn1d_confmat.txt', 'w+', encoding='utf-8') as file:
    for epoch in range(config.epochs):

        estimator.train(input_fn_train, steps=num_steps)  # Train for one epoch
        '''
        EVALUATE ON TEST DATA
        '''
        e = estimator.evaluate(input_fn_test)  # Eval after epoch

        print("Model saved in file")

        preds = [i['class_ids'] for i in estimator.predict(input_fn_test)]


        sk_acc = accuracy_score(test_y, preds)

        conf_mat = confusion_matrix(test_y, preds)

        logging.info('Testing Accuracy: %.2f | Testing Loss: %.2f | SKLearn Acc: %.2f' % (e['accuracy'], e['loss'], sk_acc))

        logging.info(str(conf_mat))

        # file.write('acc:' + str(sk_acc) + '\n' + conf_mat + '\n\n')
        # file.close()
