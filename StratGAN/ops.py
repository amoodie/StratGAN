import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



def scewl(logits, labels, name=None):
    """
    just a straight wrapper for shorter text.
    wraps: tf.nn.sigmoid_cross_entropy_with_logits()
    """
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name=name)



def relu_layer(_input, output_size, name=None, 
               stddev=0.02, bias0=0.0, return_w=False):

    _in_shape = _input.get_shape().as_list()
    # print("name", name, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(name or 'relu'):
        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.nn.relu(tf.matmul(_input, w) + b, name=name)
    
        if return_w:
            return h, w, b
        else:
            return h


def sigmoid_layer(_input, output_size, name=None, 
               stddev=0.02, bias0=0.0, return_w=False):
    _in_shape = _input.get_shape().as_list()
    # print("name", name, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(name or 'relu'):

        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.nn.sigmoid(tf.matmul(_input, w) + b, name=name)
    
        if return_w:
            return h, w, b
        else:
            return h


def linear_layer(_input, output_size, name=None, 
               stddev=0.02, bias0=0.0, return_w=False):
    _in_shape = _input.get_shape().as_list()
    # print("name", name, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(name or 'relu'):

        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.matmul(_input, w) + b
    
        if return_w:
            return h, w, b
        else:
            return h
