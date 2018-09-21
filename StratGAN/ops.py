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



def relu_layer(_input, output_size, scope=None, in_shape=None,
               stddev=0.02, bias0=0.0, return_w=False):
    
    if in_shape == None:
        _in_shape = _input.get_shape().as_list()[-1]
    else:
        _in_shape = in_shape
    print(_in_shape)

    with tf.variable_scope(scope or 'relu'):

        w = tf.get_variable("weights", [_in_shape, output_size], tf.float32,
                             initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [output_size],
                            initializer=tf.constant_initializer(bias0))
        h = tf.nn.relu(tf.matmul(_input, w) + b)
    
        if return_w:
            return h, w, b
        else:
            return h



def sigmoid_layer(_input, output_size, scope=None, 
               stddev=0.02, bias0=0.0, return_w=False):
    _in_shape = _input.get_shape().as_list()

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