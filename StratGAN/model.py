import tensorflow as tf
import numpy as np
import time
import os

import loader
import ops

from tensorflow.examples.tutorials.mnist import input_data

class StratGAN(object):
    def __init__(self, sess, config): 
        
        self.sess = sess
        self.config = config

        # Load the dataset
        if config.override_mnist:
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            self.data = mnist.train
            self.data.next_batch = tf.py_func(self.data.next_batch, [self.config.batch_size], (tf.float32, tf.float64))
            self.data.n_categories = 10
            self.data.h_dim = 28
            self.data.w_dim = 28
            self.data.c_dim = 1
            self.data.n_batches = 30000 / self.config.batch_size
        else:
            self.data = loader.ImageDatasetProvider(self.config.image_dir, 
                            image_ext=config.image_ext,             c_dim=1, 
                            batch_size=self.config.batch_size,      shuffle_data=True,
                            buffer_size=config.buffer_size,         drop_remainder=config.drop_remainder,
                            repeat_data=config.repeat_data,         a_min=None, a_max=None, 
                            verbose=config.img_verbose)

        # Initialize the net model
        self.build_model()


    def build_model(self):

        # grab some parameters for convenience:
        # -------------------
        self.y_dim = self.data.n_categories
        self.z_dim = self.config.z_dim

        
        # instantiate placeholders:
        # -------------------

        # image_dims = [self.data.h_dim, self.data.w_dim, self.data.c_dim]
        self.x = tf.placeholder(tf.float32,
                    [self.config.batch_size, self.data.h_dim * self.data.w_dim], name='x')
        self.y = tf.placeholder(tf.float32, 
                    [self.config.batch_size, self.y_dim], 
                    name='y') # labels
        self.z = tf.placeholder(tf.float32, 
                    shape=[self.config.batch_size, self.config.z_dim], 
                    name='z') # generator inputs
        self.summ_z = tf.summary.histogram('z', self.z)
        self.summ_x = tf.summary.histogram('x', self.x)


        # instantiate networks:
        # -------------------
        self.G                          = self.generator(self.z, 
                                                         self.y)
        self.D_real, self.D_real_logits = self.discriminator(self.x, 
                                                             self.y, 
                                                             reuse=False,
                                                             name="_real") # real response
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, 
                                                             self.y, 
                                                             reuse=True,
                                                             name="_fake") # fake response

        # self.G                          = self.generator(self.z, 
        #                                                  self.data.label_batch)
        # self.D_real, self.D_real_logits = self.discriminator(self.data.image_batch, 
        #                                                      self.data.label_batch, 
        #                                                      reuse=False) # real response
        # self.D_fake, self.D_fake_logits = self.discriminator(self.G, 
        #                                                      self.data.label_batch, 
        #                                                      reuse=True) # fake response
        # self.sampler = self.sampler(self.z, self.y)

        self.summ_D_real = tf.summary.histogram("D_real", self.D_real)
        self.summ_D_fake = tf.summary.histogram("D_fake", self.D_fake)
        self.summ_G = tf.summary.image("G", tf.reshape(self.G, 
                                       [self.config.batch_size, self.data.h_dim, self.data.w_dim, -1]))


        # define the losses
        # -------------------
        self.loss_d_real = tf.reduce_mean(ops.scewl(logits=self.D_real_logits, 
                                                    labels=tf.ones_like(self.D_real)), name="loss_d_real")
        self.loss_d_fake = tf.reduce_mean(ops.scewl(logits=self.D_fake_logits, 
                                                    labels=tf.zeros_like(self.D_fake)))
        self.loss_d      = self.loss_d_real + self.loss_d_fake
        self.loss_g      = tf.reduce_mean(ops.scewl(logits=self.D_fake_logits, 
                                                    labels=tf.ones_like(self.D_fake)))
        # alternative losses:
        # self.loss_d_real = tf.log(self.D_real)
        # self.loss_d_fake = tf.log(1. - self.D_fake)
        # self.loss_d      = -tf.reduce_mean(self.loss_d_real + self.loss_d_fake)
        # self.loss_g      = -tf.reduce_mean(tf.log(self.D_fake))

        self.summ_loss_d_real = tf.summary.scalar("loss_d", self.loss_d_real)
        self.summ_loss_d_fake = tf.summary.scalar("loss_d_", self.loss_d_fake)

        self.summ_loss_g = tf.summary.scalar("loss_g", self.loss_g)
        self.summ_loss_d = tf.summary.scalar("loss_d", self.loss_d)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def generator(self, z, y=None):
        out_size = self.data.w_dim * self.data.h_dim
        with tf.variable_scope('gener') as _scope:
            g_h1 = ops.relu_layer(z, out_size / 4, name='g_h1')
            g_h2 = ops.relu_layer(g_h1, out_size / 4, name='g_h2')
            g_h3 = ops.relu_layer(g_h2, out_size / 2, name='g_h3')
            g_prob = ops.sigmoid_layer(g_h3, out_size, name='g_prob')

            return g_prob


    def discriminator(self, _images, label=None, reuse=False, name=''):
        self.name = 'discr'+name
        with tf.variable_scope('discr') as scope:
            if reuse:
                scope.reuse_variables()

            # _images = tf.reshape(_images, [self.config.batch_size, 
            #                                self.data.h_dim * self.data.w_dim])

            d_h1 = ops.relu_layer(_images, 512, name='d_h1')
            d_h2 = ops.relu_layer(d_h1, 128, name='d_h2')
            d_h3 = ops.linear_layer(d_h2, 1, name='d_prob')

            return tf.nn.sigmoid(d_h2), d_h2


    def train(self):

        # solvers:
        # -------------------
        d_optim = tf.train.AdamOptimizer(self.config.learning_rate, 
                                         beta1=self.config.beta1) \
                                .minimize(self.loss_d, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.config.learning_rate, 
                                         beta1=self.config.beta1) \
                                .minimize(self.loss_g, var_list=self.g_vars)
        
        self.sess.run(tf.global_variables_initializer())
        # or? tf.global_variables_initializer().run()

        self.summ_g = tf.summary.merge([self.summ_z, self.summ_D_fake,
                                        self.summ_G, self.summ_loss_d_fake,
                                        self.summ_loss_g])
        self.summ_d = tf.summary.merge([self.summ_z, self.summ_x, self.summ_D_real, 
                                        self.summ_loss_d_real, self.summ_loss_d])
        self.writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

        cnt = 0
        start_time = time.time()
        for epoch in np.arange(self.config.epoch):
            
            # shuffle dataset
            # self.data.shuffle(self.buffer_size)

            for batch in np.arange(self.data.n_batches):
                batch_start = time.time()
                
                _image_batch, _label_batch = self.sess.run(self.data.next_batch) # have next element as the output of one shot iter
                # batch_accuracy = session.run(accuracy, feed_dict={x: images, y_true: labels, keep_prop: 1.0})
                # batch_predicted_probabilities = session.run(y_pred, feed_dict={x: images, y_true: labels, keep_prop: 1.0})

                image_batch = tf.reshape(_image_batch, [self.config.batch_size, 
                                           self.data.h_dim * self.data.w_dim]).eval()
                label_batch = _label_batch

                z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]) \
                                            .astype(np.float32)

                #### WITH FEEDDICT
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.summ_d],
                                                feed_dict={self.x: image_batch,
                                                           self.y: label_batch,
                                                           self.z: z_batch})
                self.writer.add_summary(summary_str, cnt)

                # Update G network
                for u in np.arange(self.config.g_update):
                    _, summary_str = self.sess.run([g_optim, self.summ_g],
                                                    feed_dict={self.z: z_batch,
                                                               self.y: label_batch})
                    self.writer.add_summary(summary_str, cnt)

                #### WITHOUT FEEDDICT -- DON'T KNOW HOW TO DO
                # # Update D network
                # _, summary_str = self.sess.run([d_optim, self.summ_d], feed_dict={self.z:z_batch})
                # self.writer.add_summary(summary_str, cnt)
                
                # # Update G network
                # _, summary_str = self.sess.run([g_optim, self.summ_g], feed_dict={self.z:z_batch})
                # self.writer.add_summary(summary_str, cnt)
                
                # # Update G network
                # _, summary_str = self.sess.run([g_optim, self.summ_g], feed_dict={self.z:z_batch})
                # self.writer.add_summary(summary_str, cnt)

                self.err_D_fake = self.loss_d_fake.eval({ self.z: z_batch })
                self.err_D_real = self.loss_d_real.eval({ self.x: image_batch, self.z: z_batch })
                self.err_G      = self.loss_g.eval({self.z: z_batch})

                cnt += 1
                batch_end = time.time()
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.2fs batch: %4.2fs, d_loss: %.6f, g_loss: %.6f" \
                    % (epoch, self.config.epoch, batch, self.data.n_batches,
                    time.time() - start_time, batch_end - batch_start, 
                    self.err_D_fake+self.err_D_real, self.err_G))

                # sample?

                # record chkpt
                if np.mod(cnt, 500) == 2:
                    self.saver.save(self.sess,
                        os.path.join(self.config.chkp_dir, 'StratGAN'),
                        global_step=cnt)
