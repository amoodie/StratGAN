import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



def scewl(logits, labels):
    """
    just a straight wrapper for shorter text.
    wraps: tf.nn.sigmoid_cross_entropy_with_logits()
    """
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)


def condition_concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


def condition_conv_concat(tensors, axis=3, name='conv_concat'):
    '''
    Concatenate conditioning vector on feature map axis
    '''
    if len(tensors) > 2:
        ValueError('more than 2 tensors in tensors. Only images and labels allowed.')

    x = tensors[0]
    y = tensors[1]

    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    if not axis:
        axis = tf.rank(x) + 1

    # need to create a tensor with unknown shape, little hacky here
    ones_dims = tf.stack([tf.shape(x)[0], x_shapes[1], x_shapes[2], y_shapes[3]])
    ones_tensor = tf.fill(ones_dims, 1.0)

    return condition_concat([x, y*ones_tensor], 
                            axis=3, name=name)


def conv2d_layer(_input, output_size, is_training=None, 
                 k_h=5, k_w=5, d_h=2, d_w=2, scope=None, 
                 bias0=0.0, batch_norm=False, return_w=False):
    _in_shape = _input.get_shape().as_list()

    # if batch_norm:
    #     if not is_training:
    #         RuntimeError('If batchnor, is_training MUST be passed in feeddict')

    with tf.variable_scope(scope or 'conv2d'):

        w = tf.get_variable("weights", [k_h, k_w, _in_shape[-1], output_size], tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        c = tf.nn.conv2d(_input, w, strides=[1, d_h, d_w, 1], padding='SAME')
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        
        # print("w:", w)
        # print("c:", c)
        # print("b:", b)
        m = tf.nn.bias_add(c, b)
        # print("m:", m)

        # KILLED THIS RESHAPE -- c and m had the same shape so is it needed??
        # conv = tf.reshape(m, c.get_shape())
        conv = m
        
        if batch_norm:

            print(output_size)
            # norm_shape = [output_size[1], output_size[2], output_size[3]]
            
            bn_scale = tf.get_variable("bn_scale", output_size, tf.float32,
                             initializer=tf.constant_initializer(1.0))
            bn_beta = tf.get_variable("bn_beta", output_size, tf.float32,
                             initializer=tf.constant_initializer(0.0))
            pop_mean = tf.Variable(tf.zeros(output_size), 
                                  trainable=False)
            pop_var = tf.Variable(tf.ones(output_size), 
                                 trainable=False)
            
            def training_true():
                decay = 0.95
                bn_mean, bn_var = tf.nn.moments(conv, axes=[0, 1, 2], keep_dims=False)
                train_mean = tf.assign(pop_mean,
                                      pop_mean * decay + bn_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + bn_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    bn_conv = tf.nn.batch_normalization(conv, bn_mean, bn_var,
                                                     bn_beta, bn_scale, 1e-5)
                return bn_conv

            def training_false():
                bn_conv = tf.nn.batch_normalization(conv, pop_mean, pop_var, 
                                                 bn_beta, bn_scale, 1e-5)
                return bn_conv

            bn_conv = tf.cond(is_training, true_fn=training_true, false_fn=training_false)
            h = bn_conv
        else:
            h = conv


        if return_w:
            return h, w, b
        else:
            return h



def conv2dT_layer(_input, output_size, is_training=None, 
                   k_h=5, k_w=5, d_h=2, d_w=2, scope=None, 
                   bias0=0.0, batch_norm=False, return_w=False):
    _in_shape = _input.get_shape().as_list()

    # if batch_norm:
    #     if not is_training:
    #         RuntimeError('If batchnor, is_training MUST be passed in feeddict')

    with tf.variable_scope(scope or 'deconv2d'):

        w = tf.get_variable("weights", [k_h, k_w, output_size[-1], _in_shape[-1]], tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        c = tf.nn.conv2d_transpose(_input, w, output_shape=output_size,
                                   strides=[1, d_h, d_w, 1])
        b = tf.get_variable("bias", [output_size[-1]],
                            initializer=tf.constant_initializer(bias0))
        # print("w:", w)
        # print("c:", c)
        # print("b:", b)
        m = tf.nn.bias_add(c, b)
        # print("m:", m)

        # KILLED THIS RESHAPE -- c and m had the same shape so is it needed??
        # convT = tf.reshape(m, c.get_shape())
        convT = m
        
        if batch_norm:

            norm_shape = [output_size[1], output_size[2], output_size[3]]
            
            bn_scale = tf.get_variable("bn_scale", norm_shape, tf.float32,
                             initializer=tf.constant_initializer(1.0))
            bn_beta = tf.get_variable("bn_beta", norm_shape, tf.float32,
                             initializer=tf.constant_initializer(0.0))
            pop_mean = tf.Variable(tf.zeros(norm_shape), 
                                  trainable=False)
            pop_var = tf.Variable(tf.ones(norm_shape), 
                                 trainable=False)
            
            def training_true():
                decay = 0.95
                bn_mean, bn_var = tf.nn.moments(convT, axes=[0, 1, 2], keep_dims=False)
                train_mean = tf.assign(pop_mean,
                                      pop_mean * decay + bn_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + bn_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    bn_convT = tf.nn.batch_normalization(convT, bn_mean, bn_var,
                                                     bn_beta, bn_scale, 1e-5)
                return bn_convT

            def training_false():
                bn_convT = tf.nn.batch_normalization(convT, pop_mean, pop_var, 
                                                 bn_beta, bn_scale, 1e-5)
                return bn_convT

            bn_convT = tf.cond(is_training, true_fn=training_true, false_fn=training_false)
            h = bn_convT
        else:
            h = convT


        if return_w:
            return h, w, b
        else:
            return h



def linear_layer(_input, output_size, is_training=None, scope=None, 
                 stddev=0.02, bias0=0.0, 
                 batch_norm=False, return_w=False):
    _in_shape = _input.get_shape().as_list()

    # if batch_norm:
    #     tf.cond(is_training, 
    #         true_fn=print(''), 
    #         false_fn=RuntimeError('If batchnorm, is_training MUST be passed in feeddict'))

    # print(tf.shape(_input)[1])

    with tf.variable_scope(scope or 'relu'):

        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        mm = tf.matmul(_input, w) + b
        
        if batch_norm:
            
            bn_scale = tf.get_variable("bn_scale", output_size, tf.float32,
                             initializer=tf.constant_initializer(1.0))
            bn_beta = tf.get_variable("bn_beta", output_size, tf.float32,
                             initializer=tf.constant_initializer(0.0))
            pop_mean = tf.Variable(tf.zeros(output_size), 
                                  trainable=False)
            pop_var = tf.Variable(tf.ones(output_size), 
                                 trainable=False)
            
            def training_true():
                decay = 0.95
                bn_mean, bn_var = tf.nn.moments(mm, 0, keep_dims=False)
                train_mean = tf.assign(pop_mean,
                                      pop_mean * decay + bn_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + bn_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    bn_mm = tf.nn.batch_normalization(mm, bn_mean, bn_var,
                                                     bn_beta, bn_scale, 1e-5)
                return bn_mm

            def training_false():
                bn_mm = tf.nn.batch_normalization(mm, pop_mean, pop_var, 
                                                 bn_beta, bn_scale, 1e-5)
                return bn_mm

            bn_mm = tf.cond(is_training, true_fn=training_true, false_fn=training_false)
            h = bn_mm
        else:
            h = mm


        if return_w:
            return h, w, b
        else:
            return h


def minibatch_discriminator_layer(_input, num_kernels=5, kernel_dim=3):
    """
    minibatch discrimination to prevent modal collapse:
    https://github.com/AYLIEN/gan-intro/blob/master/gan.py
    """
    # x = linear(_input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    x = linear_layer(_input, num_kernels * kernel_dim, is_training=False, scope='minibath_discrim', 
                 stddev=0.02, batch_norm=False)

    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    
    return tf.concat([_input, minibatch_features], 1)