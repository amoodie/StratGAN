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



def relu_layer(_input, output_size, scope=None, 
               stddev=0.02, bias0=0.0, return_w=False):

    _in_shape = _input.get_shape().as_list()
    # print("scope", scope, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(scope or 'relu'):
        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.nn.relu(tf.matmul(_input, w) + b)
    
        if return_w:
            return h, w, b
        else:
            return h



def leaky_relu_layer(_input, output_size, scope=None, 
                     stddev=0.02, bias0=0.0, alpha=0.2,
                     batch_norm=False, return_w=False):

    _in_shape = _input.get_shape().as_list()
    # print("scope", scope, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(scope or 'relu'):
        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        mm = tf.matmul(_input, w) + b
        
        if batch_norm:
            bn = batch_norm_op(name=scope+"_bn")
            h = tf.nn.leaky_relu(bn(mm), alpha=alpha)
        else:
            h = tf.nn.leaky_relu(mm, alpha=alpha)
    
        if return_w:
            return h, w, b
        else:
            return h



def sigmoid_layer(_input, output_size, scope=None, 
               stddev=0.02, bias0=0.0, return_w=False):
    _in_shape = _input.get_shape().as_list()
    # print("scope", scope, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(scope or 'relu'):

        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.nn.sigmoid(tf.matmul(_input, w) + b)
    
        if return_w:
            return h, w, b
        else:
            return h



def linear_layer(_input, output_size, scope=None, 
               stddev=0.02, bias0=0.0, return_w=False):
    _in_shape = _input.get_shape().as_list()
    # print("scope", scope, "in_shape:", _in_shape, "out_shape:", [_in_shape[0], output_size])

    with tf.variable_scope(scope or 'relu'):

        w = tf.get_variable("weights", [_in_shape[1], output_size], tf.float32,
                             initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.matmul(_input, w) + b
    
        if return_w:
            return h, w, b
        else:
            return h



class batch_norm_op(object):
    # copied from DCGAN
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum, 
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)
