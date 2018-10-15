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



def linear_layer(_input, output_size, is_training, scope=None, 
                 stddev=0.02, bias0=0.0, 
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
