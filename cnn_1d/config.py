class Config:
    """
    Configuration parameters
    """
    architecture_type = 'cnn_1d'  # Type of NN architecture

    # Training Parameters
    learning_rate = 0.0001  # Starting learning rate
    learning_rate_decay = True  # Learning rate decay if true
    lr_decay_steps = 1000  # Decreases learning rate after given number of steps
    lr_decay_rate = 0.96  # Learning rate decays after given amount
    dropout = 0.4  # Dropout rate
    epochs = 30  # Number of epochs
    test_size = 0.1  # Percent of data to use as training
    shuffle = True  # Shuffles if True

    # Network Parameters
    network_name = 'ConvNet1D'  # Name of network
    batch_size = 64  # Batch size
    #  IMPORTANT: filter_size and kernals must be in arrays of the same length.
    num_filters = [64, 128, 256]  # Number of filters at each conv layer
    kernel_sizes = [5, 7, 2]  # Kernel width at each conv layer
    strides = [1, 2, 1]  # Kernel strides along data at each conv layer
    padding = ['valid', 'valid', 'valid']  # type of padding at each conv layer. Alternative: 'same'
    pool_sizes = [2, 2, 2]  # Pooling widths. If None, does not pool
    # pool_sizes = [None, None, None]
    pool_strides = [1, 1, 1]  # Strides of pooling windows
    pool_padding = ['valid', 'valid', 'valid']  # type of padding at each pooling layer. Alternative: 'same'
    fully_connected_sizes = [128, 64]  # Number of nodes in fully connected/GRU lazer
    gru = False  # If True uses GRU cells instead of feed forward

    batch_norm = True  # If true, uses batch norm
    l2_regularization = False  # If true, uses l2 reg
    l2_beta = 0.01  # L2 beta value

    # Tensorflow settings
    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 1
    device_count_cpu = 1
    save_summary_steps = 4  # Saves Tensorboard scalers after given number of steps

    # Data Params
    audio_features = ['logmel']  # Type of audio processing to use
    feature_concatenation = 'append'  # If multiple audio processing, how to concatenate
