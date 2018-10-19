import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import loader
import ops
import utils


class StratGAN(object):
    def __init__(self, sess, config): 
        
        self.sess = sess
        self.config = config

        # Load the dataset
        self.data = loader.ImageDatasetProvider(self.config.image_dir, 
                                                image_ext=config.image_ext,
                                                c_dim=1, 
                                                batch_size=self.config.batch_size, 
                                                shuffle_data=True,
                                                buffer_size=config.buffer_size,
                                                drop_remainder=config.drop_remainder,
                                                repeat_data=config.repeat_data,
                                                a_min=None, a_max=None, 
                                                verbose=config.img_verbose)

        # could grab the data here if needed??
        # self.image_batch, self.label_batch = self.data.iterator.get_next()
        # self.sess.run(self.image_batch, self.label_batch)

        # print(self.image_batch)
        # grab some info from the data provider
        # self.batch_size = tf.shape(self.data.image_batch)
        self.batch_size = self.config.batch_size
        # print("config batch size:", self.batch_size)


        # batch normalization : deals with poor initialization helps gradient flow
        # copied from DCGAN
        # self.d_bn1 = batch_norm(name='d_bn1')
        # self.d_bn2 = batch_norm(name='d_bn2')
        # self.d_bn3 = batch_norm(name='d_bn3')

        # self.g_bn0 = batch_norm(name='g_bn0')
        # self.g_bn1 = batch_norm(name='g_bn1')
        # self.g_bn2 = batch_norm(name='g_bn2')
        # self.g_bn3 = batch_norm(name='g_bn3')

        # Initialize the net model
        self.build_model()


    def build_model(self):

        # grab some parameters for convenience:
        # -------------------
        self.y_dim = self.data.n_categories
        self.z_dim = self.config.z_dim
        self.c_dim = self.data.c_dim

        
        # instantiate placeholders:
        # -------------------
        self.x = tf.placeholder(tf.float32,
                    [None, self.data.h_dim, self.data.w_dim, self.data.c_dim],
                    name='x')
        self.y = tf.placeholder(tf.float32, 
                    [None, self.y_dim], 
                    name='y') # labels
        self.z = tf.placeholder(tf.float32, 
                    shape=[None, self.config.z_dim], 
                    name='z') # generator inputs
        self.encoded = tf.placeholder(tf.int8, 
                          shape=[None, self.data.n_categories], 
                          name='encoded') # generator inputs
        self.is_training = tf.placeholder(tf.bool, name='is_training')




        self.summ_z = tf.summary.histogram('z', self.z)


        # instantiate networks:
        # -------------------
        self.G                          = self.generator(_z=self.z, 
                                                         _labels=self.y,
                                                         is_training=self.is_training,
                                                         batch_norm=self.config.batch_norm)
        self.D_real, self.D_real_logits = self.discriminator(self.x, 
                                                             self.y, 
                                                             reuse=False,
                                                             is_training=self.is_training,
                                                             batch_norm=self.config.batch_norm) # real response
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, 
                                                             self.y, 
                                                             reuse=True,
                                                             is_training=self.is_training,
                                                             batch_norm=self.config.batch_norm) # fake response
        # self.sampler = self.sampler(self.z, self.y)

        self.summ_D_real = tf.summary.histogram("D_real", self.D_real)
        self.summ_D_fake = tf.summary.histogram("D_fake", self.D_fake)
        self.summ_G = tf.summary.image("G", tf.reshape(self.G, 
                                       [self.batch_size, self.data.h_dim, self.data.w_dim, -1]))


        # define the losses
        # -------------------
        self.loss_d_real = tf.reduce_mean(ops.scewl(logits=self.D_real_logits, 
                                                    labels=tf.ones_like(self.D_real)))
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

        self.summ_loss_g = tf.summary.scalar("loss_g", self.loss_g)
        self.summ_loss_d = tf.summary.scalar("loss_d", self.loss_d)

        self.summ_loss_d_real = tf.summary.scalar("loss_d_real", self.loss_d_real)
        self.summ_loss_d_fake = tf.summary.scalar("loss_d_fake", self.loss_d_fake)

        self.summ_image = tf.summary.histogram("images", self.x)
        self.summ_label = tf.summary.histogram("labels", self.y)
        self.summ_z     = tf.summary.histogram("zs", self.z)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def generator(self, _z, _labels, is_training=False, batch_norm=False):
        
        _out_size = self.data.w_dim * self.data.h_dim

        _batch_size = tf.shape(_z)[0] # dynamic batch size op
        
        with tf.control_dependencies([_batch_size]):

            # print("batch_size:", _batch_size)

            with tf.variable_scope('gener') as scope:
                s_h, s_w = self.data.h_dim, self.data.w_dim
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                # reshape the labels for concatenation to feature axis of conv tensors
                _labels_r = tf.reshape(_labels, [_batch_size, 1, 1, self.y_dim])

                # fully connected, layer 0
                g_c0 = ops.condition_concat([_z, _labels], axis=1, name='g_cat0')
                g_h0 = ops.linear_layer(g_c0, self.config.gfc_dim, 
                                        is_training=is_training, 
                                        scope='g_h0', batch_norm=batch_norm)
                g_h0 = tf.nn.relu(g_h0)

                # fully connected, layer 1
                g_c1 = ops.condition_concat([g_h0, _labels], axis=1, name='g_cat1')
                g_h1 = ops.linear_layer(g_c1, self.config.gf_dim*2*s_h4*s_w4, 
                                        is_training=is_training, 
                                        scope='g_h1', batch_norm=batch_norm)
                g_h1 = tf.nn.relu(g_h1)

                # deconvolution, layer 2
                g_r2 = tf.reshape(g_h1, [_batch_size, s_h4, s_w4, self.config.gf_dim * 2])
                g_c2 = ops.condition_conv_concat([g_r2, _labels_r], axis=3, 
                                                 name='g_cat2')
                g_h2 = ops.conv2dT_layer(g_c2, [_batch_size, s_h2, s_w2, self.config.gf_dim * 2],
                                         is_training=is_training, 
                                         scope='g_h2', batch_norm=batch_norm)
                g_h2 = tf.nn.relu(g_h2)

                # deconvolution, layer 3
                g_c3 = ops.condition_conv_concat([g_h2, _labels_r], axis=3, 
                                                 name='g_cat3')
                g_h3 = ops.conv2dT_layer(g_c3, [_batch_size, s_h, s_w, self.data.c_dim],
                                         is_training=is_training, 
                                         scope='g_h3', batch_norm=False)
                g_prob = tf.nn.sigmoid(g_h3)

                return g_prob


    def discriminator(self, _images, _labels,
                      reuse=False, batch_norm=False, is_training=True):
        
        _in_size = self.data.w_dim * self.data.h_dim

        _batch_size = tf.shape(_images)[0] # dynamic batch size op
            
        with tf.control_dependencies([_batch_size]):

            with tf.variable_scope('discr') as scope:
                if reuse:
                    scope.reuse_variables()

                # reshape the labels for concatenation to feature axis of conv tensors
                _labels_r = tf.reshape(_labels, [-1, 1, 1, self.y_dim])

                # convolution, layer 0
                # print("cdim:",self.data.c_dim)
                # print("ydim:",self.data.y_dim)
                # print("cdim+ydim:",self.data.c_dim + self.data.y_dim)
                # print("_labels:",_labels)
                # print("_labels_r:",_labels_r)

                d_c0 = ops.condition_conv_concat([_images, _labels_r], axis=3, 
                                                 name='d_cat0')
                # print("d_c0:", d_c0)
                d_h0 = ops.conv2d_layer(d_c0, self.data.c_dim + self.data.y_dim, 
                                        is_training=False, 
                                        k_h=5, k_w=5, d_h=2, d_w=2, 
                                        scope='d_h0', batch_norm=False)
                d_h0 = tf.nn.leaky_relu(d_h0, alpha=self.config.alpha)

                # convolution, layer 1
                d_c1 = ops.condition_conv_concat([d_h0, _labels_r], axis=3,
                                                 name='d_cat1')
                d_h1 = ops.conv2d_layer(d_c1, self.config.df_dim + self.y_dim, 
                                        is_training=is_training, 
                                        k_h=5, k_w=5, d_h=2, d_w=2, 
                                        scope='d_h1', batch_norm=batch_norm)
                d_h1 = tf.nn.leaky_relu(d_h1, alpha=self.config.alpha)
                # print("d_h1:",d_h1)

                # fully connected, layer 2
                d_r2 = tf.reshape(d_h1, [_batch_size, 7*7*68])
                d_c2 = ops.condition_concat([d_r2, _labels], axis=1, 
                                            name='d_cat2')
                # print('dc2:', d_c2)
                d_h2 = ops.linear_layer(d_c2, self.config.dfc_dim, 
                                        is_training=is_training, 
                                        scope='d_h2', batch_norm=batch_norm)
                d_h2 = tf.nn.leaky_relu(d_h2, alpha=self.config.alpha)

                # fully connected, layer 3
                d_c3 = ops.condition_concat([d_h2, _labels], axis=1, 
                                            name='d_cat3')
                d_h3 = ops.linear_layer(d_c3, 1, 
                                        is_training=False, 
                                        scope='d_h3', batch_norm=False)
                d_prob = tf.nn.sigmoid(d_h3)

                return d_prob, d_h3


    def train(self):

        # solvers:
        # -------------------
        d_optim = tf.train.AdamOptimizer(self.config.learning_rate, 
                                         beta1=self.config.beta1) \
                                .minimize(self.loss_d, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.config.learning_rate, 
                                         beta1=self.config.beta1) \
                                .minimize(self.loss_g, var_list=self.g_vars)
        # training_true = tf.constant(True, dtype=tf.bool)
        # training_false = tf.constant(False, dtype=tf.bool)
        
        self.sess.run(tf.global_variables_initializer())

        self.summ_g = tf.summary.merge([self.summ_D_fake, self.summ_G, 
                                        self.summ_loss_d_fake, self.summ_loss_g])
        self.summ_d = tf.summary.merge([self.summ_D_real, self.summ_loss_d_real, 
                                        self.summ_loss_d])
        self.summ_input = tf.summary.merge([self.summ_image, self.summ_label, 
                                            self.summ_z])
        self.writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

        self.decoder = tf.argmax(self.encoded, axis=1)

        self.training_zs, self.training_labels = utils.training_sample_set(
                                                        self.config.z_dim, 
                                                        self.data.n_categories)

        cnt = 0
        start_time = time.time()
        print("Start time: ", start_time)
        for epoch in np.arange(self.config.epoch):
            
            # shuffle dataset
            # self.data.shuffle(self.buffer_size)

            for batch in np.arange(self.data.n_batches):
                
                _image_batch, _label_batch = self.sess.run(self.data.next_batch)

                z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]) \
                                            .astype(np.float32)

                # image_batch = tf.reshape(_image_batch, [self.config.batch_size, 
                #                            self.data.h_dim * self.data.w_dim]).eval()
                image_batch = _image_batch.copy()

                
                label_batch = _label_batch.copy()

                if self.config.noisy_inputs:
                    image_batch = image_batch + 1 * np.random.normal(0, 0.1, size=image_batch.shape)
                if self.config.flip_inputs:
                    image_batch = 1 - image_batch

                # run for new inputs data (slow?)
                # summary_str = self.sess.run(self.summ_input, 
                #                             feed_dict={self.x: image_batch,
                #                                        self.y: label_batch,
                #                                        self.z: z_batch})
                # self.writer.add_summary(summary_str, cnt)


                # update networks:
                # -------------------

                # update D network
                _, summary_str = self.sess.run([d_optim, self.summ_d], 
                                                feed_dict={self.x: image_batch,
                                                           self.y: label_batch,
                                                           self.z: z_batch,
                                                           self.is_training: True})
                # self.writer.add_summary(summary_str, cnt)
                
                # update G network
                for g in np.arange(self.config.gener_iter):
                    _, summary_str = self.sess.run([g_optim, self.summ_g], 
                                                    feed_dict={self.z: z_batch,
                                                               self.y: label_batch,
                                                               self.is_training: True})
                # self.writer.add_summary(summary_str, cnt)


                self.err_D_fake = self.loss_d_fake.eval({ self.z: z_batch, 
                                                          self.y: label_batch,
                                                          self.is_training: False })
                self.err_D_real = self.loss_d_real.eval({ self.x: image_batch, 
                                                          self.y: label_batch,
                                                          self.is_training: False })
                self.err_G      = self.loss_g.eval({ self.z: z_batch, 
                                                     self.y: label_batch,
                                                     self.is_training: False })


                if cnt % 25 == 0:
                    self.sampler(self.training_zs, _labels=self.training_labels, 
                                 time=[epoch, batch])

                    # make plot of input images:
                    # -------------------
                    # fig = utils.plot_images(image_batch[:16, ...], 
                    #                         dim=self.data.h_dim, 
                    #                         labels=decoded[:16, ...])
                    # plt.savefig('out/x_{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
                    # plt.close(fig)
                    pass

                cnt += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.6f, g_loss: %.6f" \
                    % (epoch+1, self.config.epoch, batch+1, self.data.n_batches,
                    time.time() - start_time, self.err_D_fake+self.err_D_real, self.err_G))

                # record chkpt
                # if np.mod(cnt, 500) == 2:
                #     self.saver.save(self.sess,
                #         os.path.join(self.config.chkp_dir, 'StratGAN'),
                #         global_step=cnt)

    def sampler(self, z, _labels=None, time=None):
        
        epoch = time[0]
        batch = time[1]

        samples, decoded = self.sess.run([self.G, self.decoder], 
                                          feed_dict={self.z: z, 
                                                     self.y: _labels,
                                                     self.encoded: _labels,
                                                     self.is_training: False})
        fig = utils.plot_images(samples, image_dim=self.data.h_dim, 
                                n_categories=self.data.n_categories, 
                                labels=decoded)
        file_name = 'samp/g_{0}_{1}.png'.format(str(epoch+1).zfill(3), 
                                                str(batch).zfill(4))
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)
        print("Sample: {file_name}".format(file_name=file_name))
