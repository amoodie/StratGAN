import tensorflow as tf
import numpy as np

import loader


class StratGAN(object):
    def __init__(self, sess, config): 
        
        self.sess = sess
        self.config = config

        # Load the dataset
        self.data = loader.ImageDatasetProvider(self.config.image_dir, 
                                                image_ext=config.image_ext,
                                                c_dim=1, 
                                                batch_size=self.config.batch_size, 
                                                shuffle_data=True, buffer_size=config.buffer_size,
                                                repeat_data=config.repeat_data,
                                                a_min=None, a_max=None, 
                                                verbose=config.img_verbose)


    def __fakeinit__(self):
        X = tf.placeholder(tf.float32, shape=[None, 784])

        D_W1 = tf.Variable(xavier_init([784, 128]))
        D_b1 = tf.Variable(tf.zeros(shape=[128]))

        D_W2 = tf.Variable(xavier_init([128, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_W1, D_W2, D_b1, D_b2]


        Z = tf.placeholder(tf.float32, shape=[None, 100])

        G_W1 = tf.Variable(xavier_init([100, 128]))
        G_b1 = tf.Variable(tf.zeros(shape=[128]))

        G_W2 = tf.Variable(xavier_init([128, 784]))
        G_b2 = tf.Variable(tf.zeros(shape=[784]))

        theta_G = [G_W1, G_W2, G_b1, G_b2]

        # instantiate networks:
        # -------------------
        G_sample = generator(Z)
        D_real, D_logit_real = discriminator(X)
        D_fake, D_logit_fake = discriminator(G_sample)

        # alternative losses:
        # -------------------
        # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        # G_loss = -tf.reduce_mean(tf.log(D_fake))
        
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


        # solver:
        # -------------------
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


    def generator(z, y=None):
        with tf.variable_scope("gener") as scope:
            G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)

            return G_prob


    def discriminator(x):
        with tf.variable_scope("discr") as scope:
            D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
            D_logit = tf.matmul(D_h1, D_W2) + D_b2
            D_prob = tf.nn.sigmoid(D_logit)

            return D_prob, D_logit

    def train(self, config):
        
        sess.run(tf.global_variables_initializer())

        # load up the dataset (all paths and labels?)

        cnt = 0
        start_time = time.time()
        for epoch in np.arange(config.epoch):
            
            # shuffle dataset

            for batch in np.arange(num_batches):
                # sess.run update D
                # sess.run update G
                # _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
                # _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

                # print some summary stats?

                # sample?

                pass


            # sample
            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()


            if it % 1000 == 0:
                samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)

            pass