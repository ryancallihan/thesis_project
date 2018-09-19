class Config:
    """
    Configuration parameters
    """
    architecture_type = 'rnn'  # Type of NN architecture

    # Training Parameters
    learning_rate = 0.001  # Learning rate for optimiser
    learning_rate_decay = True  # Learning rate decay if true
    lr_decay_steps = 1000  # Decreases learning rate after given number of steps
    lr_decay_rate = 0.96  # Learning rate decays after given amount
    dropout = 0.2  # Blanket dropout for all training layers.
    epochs = 30  # Number of epochs
    test_size = 0.2  # % of data set aside for training.
    shuffle = True  # If using time series (RNN), must be false.

    # Network Parameters
    batch_size = 64  # Batch size
    hidden_sizes = [256, 256, 64]  # Number of neurons per hidden layer. e.g. [256, 64] = 2 hidden layers
    bidirectional = False  # If you want a bidirectional GRU RNN. Takes much longer to train.
    cell_type = 'gru'  # 'gru' 'rnn'

    l2_regularization = True  # If true, uses l2 reg
    l2_beta = 0.01  # L2 beta value
    batch_norm = False  # If true, uses batch norm

    # Tensorflow settings
    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 1
    device_count_cpu = 1
    save_summary_steps = 4

    # Data Params
    audio_features = ['mfcc']  # Type of audio processing to use
    # frame_size = 0.025  # 0.025 = 25 ms
    # frame_overlap = 0.01  # 0.01 = 10 ms
