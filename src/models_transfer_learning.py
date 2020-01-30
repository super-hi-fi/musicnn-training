import tensorflow as tf
# from musicnn import configuration as config

# disabling deprecation warnings (caused by change from tensorflow 1.x to 2.x)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def define_model(x, is_training, model_num, num_classes):
    if model_num == 2:
        model = 'MTT_vgg'
    elif model_num == 11:
        model = 'MTT_musicnn'
    elif model_num == 20:
        model = 'audioset_vgg'
    elif model_num == 21:
        model = 'small_vggish'
    else:
        raise Exception('model number not contemplated for transfer learning')

    if model == 'MTT_musicnn':
        return build_musicnn(x, is_training, num_classes, num_filt_midend=64, num_units_backend=200)

    elif model == 'MTT_vgg':
        return vgg(x, is_training, num_classes, 128)

    elif model == 'MSD_musicnn':
        return build_musicnn(x, is_training, num_classes, num_filt_midend=64, num_units_backend=200)

    elif model == 'MSD_musicnn_big':
        return build_musicnn(x, is_training, num_classes, num_filt_midend=512, num_units_backend=500)

    elif model == 'MSD_vgg':
        return vgg(x, is_training, num_classes, 128)

    elif model == 'audioset_vgg':
        return define_vggish_slim(x, is_training, num_classes)

    elif model == 'small_vggish':
        return define_small_vggish_slim(x, is_training, num_classes)

    else:
        raise ValueError('Model not implemented!')


def build_musicnn(x, is_training, num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200):

    ### front-end ### musically motivated CNN
    non_trainable = False
    frontend_features_list = frontend(x, non_trainable, 96, num_filt=1.6, type='7774timbraltemporal')
    # concatnate features coming from the front-end
    frontend_features = tf.concat(frontend_features_list, 2)

    ### mid-end ### dense layers
    midend_features_list = midend(frontend_features, non_trainable, num_filt_midend)
    # dense connection: concatnate features coming from different layers of the front- and mid-end
    midend_features = tf.concat(midend_features_list, 2)

    ### back-end ### temporal pooling
    logits, penultimate, mean_pool, max_pool = backend(midend_features, is_training, num_classes, num_units_backend, type='globalpool_dense')

    # [extract features] temporal and timbral features from the front-end
    timbral = tf.concat([frontend_features_list[0], frontend_features_list[1]], 2)
    temporal = tf.concat([frontend_features_list[2], frontend_features_list[3], frontend_features_list[4]], 2)
    # [extract features] mid-end features
    cnn1, cnn2, cnn3 = midend_features_list[1], midend_features_list[2], midend_features_list[3]
    mean_pool = tf.squeeze(mean_pool, [2])
    max_pool = tf.squeeze(max_pool, [2])

    return logits


def frontend(x, is_training, yInput, num_filt, type):

    expand_input = tf.expand_dims(x, 3)
    normalized_input = tf.compat.v1.layers.batch_normalization(expand_input, training=is_training)

    if 'timbral' in type:

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(normalized_input, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        if '74' in type:
            f74 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.4 * yInput)],
                           is_training=is_training)

        if '77' in type:
            f77 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.7 * yInput)],
                           is_training=is_training)

    if 'temporal' in type:

        s1 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=[128,1],
                          is_training=is_training)

        s2 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=[64,1],
                          is_training=is_training)

        s3 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=[32,1],
                          is_training=is_training)


    # choose the feature maps we want to use for the experiment
    if type == '7774timbraltemporal':
        return [f74, f77, s1, s2, s3]


def timbral_block(inputs, filters, kernel_size, is_training, padding="valid", activation=tf.nn.relu):

    conv = tf.compat.v1.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            activation=activation)
    bn_conv = tf.compat.v1.layers.batch_normalization(conv, training=is_training)
    pool = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv,
                                   pool_size=[1, bn_conv.shape[2]],
                                   strides=[1, bn_conv.shape[2]])
    return tf.squeeze(pool, [2])


def tempo_block(inputs, filters, kernel_size, is_training, padding="same", activation=tf.nn.relu):

    conv = tf.compat.v1.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            activation=activation)
    bn_conv = tf.compat.v1.layers.batch_normalization(conv, training=is_training)
    pool = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv,
                                   pool_size=[1, bn_conv.shape[2]],
                                   strides=[1, bn_conv.shape[2]])
    return tf.squeeze(pool, [2])


def midend(front_end_output, is_training, num_filt):

    front_end_output = tf.expand_dims(front_end_output, 3)

    # conv layer 1 - adapting dimensions
    front_end_pad = tf.pad(front_end_output, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv1 = tf.compat.v1.layers.conv2d(inputs=front_end_pad,
                             filters=num_filt,
                             kernel_size=[7, front_end_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu)
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.compat.v1.layers.conv2d(inputs=bn_conv1_pad,
                             filters=num_filt,
                             kernel_size=[7, bn_conv1_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu)
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.transpose(bn_conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2, bn_conv1_t)

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv3 = tf.compat.v1.layers.conv2d(inputs=bn_conv2_pad,
                             filters=num_filt,
                             kernel_size=[7, bn_conv2_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu)
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.transpose(bn_conv3, [0, 1, 3, 2])
    res_conv3 = tf.add(conv3, res_conv2)

    return [front_end_output, bn_conv1_t, res_conv2, res_conv3]


def backend(feature_map, is_training, num_classes, output_units, type):

    # temporal pooling
    max_pool = tf.reduce_max(feature_map, axis=1)
    mean_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
    tmp_pool = tf.concat([max_pool, mean_pool], 2)

    # penultimate dense layer
    flat_pool = tf.compat.v1.layers.flatten(tmp_pool)
    flat_pool = tf.compat.v1.layers.batch_normalization(flat_pool, training=False)
    flat_pool_dropout = tf.compat.v1.layers.dropout(flat_pool, rate=0.5, training=False)
    dense = tf.compat.v1.layers.dense(inputs=flat_pool_dropout,
                            units=output_units,
                            activation=tf.nn.relu)
    bn_dense = tf.compat.v1.layers.batch_normalization(dense, training=False)
    dense_dropout = tf.compat.v1.layers.dropout(bn_dense, rate=0.5, training=False)

    # output dense layer
    ld = tf.compat.v1.layers.dense(inputs=dense_dropout,
                           activation=None,
                           units=100)
    logits = tf.compat.v1.layers.dense(inputs=ld,
                           activation=None,
                           units=num_classes)

    return logits, bn_dense, mean_pool, max_pool


def vgg(x, is_training, num_classes, num_filters=32):
    non_trainable = False

    input_layer = tf.expand_dims(x, 3)
    bn_input = tf.compat.v1.layers.batch_normalization(input_layer, training=non_trainable)

    conv1 = tf.compat.v1.layers.conv2d(inputs=bn_input,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='1CNN')
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=non_trainable)
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv1, pool_size=[4, 1], strides=[2, 2])

    do_pool1 = tf.compat.v1.layers.dropout(pool1, rate=0.25, training=non_trainable)
    conv2 = tf.compat.v1.layers.conv2d(inputs=do_pool1,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='2CNN')
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=non_trainable)
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv2, pool_size=[2, 2], strides=[2, 2])

    do_pool2 = tf.compat.v1.layers.dropout(pool2, rate=0.25, training=non_trainable)
    conv3 = tf.compat.v1.layers.conv2d(inputs=do_pool2,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='3CNN')
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=non_trainable)
    pool3 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv3, pool_size=[2, 2], strides=[2, 2])

    do_pool3 = tf.compat.v1.layers.dropout(pool3, rate=0.25, training=non_trainable)
    conv4 = tf.compat.v1.layers.conv2d(inputs=do_pool3,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='4CNN')
    bn_conv4 = tf.compat.v1.layers.batch_normalization(conv4, training=non_trainable)
    pool4 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv4, pool_size=[2, 2], strides=[2, 2])

    do_pool4 = tf.compat.v1.layers.dropout(pool4, rate=0.25, training=non_trainable)
    conv5 = tf.compat.v1.layers.conv2d(inputs=do_pool4,
                             filters=num_filters, 
                             kernel_size=[3, 3], 
                             padding='same', 
                             activation=tf.nn.relu,
                             name='5CNN')
    bn_conv5 = tf.compat.v1.layers.batch_normalization(conv5, training=is_training)
    pool5 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv5, pool_size=[4, 4], strides=[4, 4])

    flat_pool5 = tf.compat.v1.layers.flatten(pool5)
    do_pool5 = tf.compat.v1.layers.dropout(flat_pool5, rate=0.5, training=is_training)
    output = tf.compat.v1.layers.dense(inputs=do_pool5,
                            activation=None,
                            units=num_classes)
    return output


def define_vggish_slim(x, is_training, num_classes):
    """Defines the VGGish TensorFlow model.
    All ops are created in the current default graph, under the scope 'vggish/'.
    The input is a placeholder named 'vggish/input_features' of type float32 and
    shape [batch_size, num_frames, num_bands] where batch_size is variable and
    num_frames and num_bands are constants, and [num_frames, num_bands] represents
    a log-mel-scale spectrogram patch covering num_bands frequency bands and
    num_frames time frames (where each frame step is usually 10ms). This is
    produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).
    The output is an op named 'vggish/embedding' which produces the activations of
    a 128-D embedding layer, which is usually the penultimate layer when used as
    part of a full model with a final classifier layer.
    Args:
    training: If true, all parameters are marked trainable.
    Returns:
    The op 'vggish/embeddings'.
    """

    slim = tf.contrib.slim

    # Architectural constants.
    EMBEDDING_SIZE = 128  # Size of embedding layer.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.

    # Defaults:
    # - All weights are initialized to N(0, INIT_STDDEV).
    # - All biases are initialized to 0.
    # - All activations are ReLU.
    # - All convolutions are 3x3 with stride 1 and SAME padding.
    # - All max-pools are 2x2 with stride 2 and SAME padding.

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=INIT_STDDEV),
                        biases_initializer=tf.zeros_initializer(),
                        activation_fn=tf.nn.relu,
                        trainable=False), \
        slim.arg_scope([slim.conv2d],
                        kernel_size=[3, 3], stride=1, padding='SAME'), \
        slim.arg_scope([slim.max_pool2d],
                        kernel_size=[2, 2], stride=2, padding='SAME'), \
        tf.variable_scope('vggish'):

        # Reshape to 4-D so that we can convolve a batch with conv2d().
        net = tf.reshape(x, [-1, NUM_FRAMES, NUM_BANDS, 1])

        # The VGG stack of alternating convolutions and max-pools.
        net = slim.conv2d(net, 64, scope='conv1')
        net = slim.max_pool2d(net, scope='pool1')
        net = slim.conv2d(net, 128, scope='conv2')
        net = slim.max_pool2d(net, scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
        net = slim.max_pool2d(net, scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
        net = slim.max_pool2d(net, scope='pool4')

        # Flatten before entering fully-connected layers
        net = slim.flatten(net)
        net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
        # The embedding layer.
        embeddings = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc2')

    num_units = 100
    fc = slim.fully_connected(embeddings, num_units)

    # Add a classifier layer at the end, consisting of parallel logistic
    # classifiers, one per class. This allows for multi-class tasks.
    logits = slim.fully_connected(
        fc, num_classes, activation_fn=None, scope='logits')
    tf.sigmoid(logits, name='prediction')

    return tf.identity(logits, name='logits')


def define_small_vggish_slim(x, is_training, num_classes):
    """Defines a small VGGish TensorFlow model.
    WARNING: THIS MODEL IS NOT TRAINED.

    The number of filters and fully-conected units are divided by 64 and the
    weights are randomly initialized.

    This models is created for testing purposes only due to the huge size of the
    full VGGish. (~165k vs. ~280MB once serialized.)

    Args:
    training: If true, all parameters are marked trainable.
    Returns:
    The op 'vggish/embeddings'.
    """

    slim = tf.contrib.slim

    # Architectural constants.
    EMBEDDING_SIZE = 128  # Size of embedding layer.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.

    # Defaults:
    # - All weights are initialized to N(0, INIT_STDDEV).
    # - All biases are initialized to 0.
    # - All activations are ReLU.
    # - All convolutions are 3x3 with stride 1 and SAME padding.
    # - All max-pools are 2x2 with stride 2 and SAME padding.

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=INIT_STDDEV),
                        biases_initializer=tf.zeros_initializer(),
                        activation_fn=tf.nn.relu,
                        trainable=False), \
        slim.arg_scope([slim.conv2d],
                        kernel_size=[3, 3], stride=1, padding='SAME'), \
        slim.arg_scope([slim.max_pool2d],
                        kernel_size=[2, 2], stride=2, padding='SAME'), \
        tf.variable_scope('vggish'):

        # Reshape to 4-D so that we can convolve a batch with conv2d().
        net = tf.reshape(x, [-1, NUM_FRAMES, NUM_BANDS, 1])

        # The VGG stack of alternating convolutions and max-pools.
        net = slim.conv2d(net, 1, scope='conv1')
        net = slim.max_pool2d(net, scope='pool1')
        net = slim.conv2d(net, 2, scope='conv2')
        net = slim.max_pool2d(net, scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 4, scope='conv3')
        net = slim.max_pool2d(net, scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 8, scope='conv4')
        net = slim.max_pool2d(net, scope='pool4')

        # Flatten before entering fully-connected layers
        net = slim.flatten(net)
        net = slim.repeat(net, 2, slim.fully_connected, 64, scope='fc1')
        # The embedding layer.
        embeddings = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc2')

    num_units = 100
    fc = slim.fully_connected(embeddings, num_units)

    # Add a classifier layer at the end, consisting of parallel logistic
    # classifiers, one per class. This allows for multi-class tasks.
    logits = slim.fully_connected(
        fc, num_classes, activation_fn=None, scope='logits')
    tf.sigmoid(logits, name='prediction')

    return tf.identity(logits, name='logits')
