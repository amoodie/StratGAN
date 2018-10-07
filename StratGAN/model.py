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
        self.i_dim = self.data.h_dim * self.data.w_dim # image dimensions

        
        # instantiate placeholders:
        # -------------------
        self.x = tf.placeholder(tf.float32,
                    [None, self.data.h_dim * self.data.w_dim], name='x')
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
        self.G                          = self.generator(z=self.z, 
                                                         labels=self.y,
                                                         is_training=self.is_training,
                                                         batch_norm=self.config.batch_norm)
        
        # make two mixed batches of fake and real examples for the discriminator
        self.xG1 = tf.concat([tf.slice(self.x, [0, 0], 
                                       [self.config.batch_size//2, self.data.h_dim * self.data.w_dim]),
                              tf.slice(self.G, [0, 0], 
                                       [self.config.batch_size//2, self.data.h_dim * self.data.w_dim])], 
                             axis=0) # 50/50 part 1
        self.xG2 = tf.concat([tf.slice(self.x, [self.config.batch_size//2, 0], 
                                       [self.config.batch_size//2, self.data.h_dim * self.data.w_dim]),
                              tf.slice(self.G, [self.config.batch_size//2, 0], 
                                       [self.config.batch_size//2, self.data.h_dim * self.data.w_dim])], 
                             axis=0)# 50/50 part 2
        self.y1 = tf.concat([tf.slice(self.y, [0, 0], 
                                       [self.config.batch_size//2, self.data.n_categories]),
                             tf.slice(self.y, [0, 0], 
                                       [self.config.batch_size//2, self.data.n_categories])], 
                             axis=0) # 50/50 part 1
        self.y2 = tf.concat([tf.slice(self.y, [self.config.batch_size//2, 0], 
                                       [self.config.batch_size//2, self.data.n_categories]),
                              tf.slice(self.y, [self.config.batch_size//2, 0], 
                                       [self.config.batch_size//2, self.data.n_categories])], 
                             axis=0)# 50/50 part 2

        self.D_real, self.D_real_logits = self.discriminator(self.xG1, 
                                                             self.y1, 
                                                             reuse=False,
                                                             batch_norm=self.config.batch_norm) # real response
        self.D_fake, self.D_fake_logits = self.discriminator(self.xG2, 
                                                             self.y2, 
                                                             reuse=True,
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


    def generator(self, z, labels, is_training=False, batch_norm=False):
        
        _out_size = self.data.w_dim * self.data.h_dim

        with tf.variable_scope('gener') as scope:
        
            g_c1 = tf.concat([z, labels], axis=1, name='g_c1')
            g_h1 = ops.leaky_relu_layer(g_c1, _out_size // 4,
                                        scope='g_h1', batch_norm=batch_norm,
                                        is_training=is_training)

            g_c2 = tf.concat([g_h1, labels], axis=1, name='g_c2')
            g_h2 = ops.leaky_relu_layer(g_c2, _out_size // 4,
                                        scope='g_h2', batch_norm=batch_norm,
                                        is_training=is_training)
            
            g_c3 = tf.concat([g_h2, labels], axis=1, name='g_c3')
            g_h3 = ops.leaky_relu_layer(g_c3, _out_size // 3,
                                        scope='g_h3', batch_norm=batch_norm,
                                        is_training=is_training)
            
            g_c4 = tf.concat([g_h3, labels], axis=1, name='g_c4')
            g_h4 = ops.leaky_relu_layer(g_c4, _out_size // 2,
                                        scope='g_h4', batch_norm=batch_norm,
                                        is_training=is_training)
            
            g_c5 = tf.concat([g_h4, labels], axis=1, name='g_c5')
            g_prob = ops.sigmoid_layer(g_c5, _out_size,
                                        scope='g_h5', batch_norm=False,
                                        is_training=is_training)

            return g_prob


    def discriminator(self, _images, labels,
                      reuse=False, batch_norm=False, is_training=True):
        
        _in_size = self.data.w_dim * self.data.h_dim
        training_true = tf.constant(True, dtype=tf.bool)

        with tf.variable_scope('discr') as scope:
            if reuse:
                scope.reuse_variables()

            d_c1 = tf.concat([_images, labels], axis=1, name='d_c1')
            d_h1 = ops.leaky_relu_layer(d_c1, _in_size // 2, 
                                        scope='d_h1', batch_norm=batch_norm,
                                        is_training=training_true)

            d_c2 = tf.concat([d_h1, labels], axis=1, name='d_c2')
            d_h2 = ops.leaky_relu_layer(d_c2, _in_size // 4, 
                                        scope='d_h2', batch_norm=batch_norm,
                                        is_training=training_true)

            d_c3 = tf.concat([d_h2, labels], axis=1, name='d_c3')
            d_h3 = ops.leaky_relu_layer(d_c3, _in_size // 8, 
                                        scope='d_h3', batch_norm=batch_norm,
                                        is_training=training_true)

            d_c4 = tf.concat([d_h3, labels], axis=1, name='d_c4')
            d_h4 = ops.linear_layer(d_c4, 1, 
                                    scope='d_h4', batch_norm=False,
                                    is_training=training_true)

            d_prob = tf.nn.sigmoid(d_h4)

            return d_prob, d_h4


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
        for epoch in np.arange(self.config.epoch):
            
            # shuffle dataset
            # self.data.shuffle(self.buffer_size)

            for batch in np.arange(self.data.n_batches):
                
                _image_batch, _label_batch = self.sess.run(self.data.next_batch)

                z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]) \
                                            .astype(np.float32)

                image_batch = tf.reshape(_image_batch, [self.config.batch_size, 
                                           self.data.h_dim * self.data.w_dim]).eval()
                
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
                self.writer.add_summary(summary_str, cnt)
                
                # update G network
                for g in np.arange(self.config.gener_iter):
                    _, summary_str = self.sess.run([g_optim, self.summ_g], 
                                                    feed_dict={self.z: z_batch,
                                                               self.x: image_batch,
                                                               self.y: label_batch,
                                                               self.is_training: True})
                self.writer.add_summary(summary_str, cnt)


                self.err_D_fake = self.loss_d_fake.eval({ self.z: z_batch, 
                                                          self.x: image_batch,
                                                          self.y: label_batch,
                                                          self.is_training: False })
                self.err_D_real = self.loss_d_real.eval({ self.z: z_batch, 
                                                          self.x: image_batch, 
                                                          self.y: label_batch,
                                                          self.is_training: False })
                self.err_G      = self.loss_g.eval({ self.z: z_batch, 
                                                     self.x: image_batch,
                                                     self.y: label_batch,
                                                     self.is_training: False })


                if cnt % 25 == 0:
                    self.sampler(self.training_zs, image_batch, _labels=self.training_labels, 
                                 time=[epoch, batch])

                    # make plot of input images:
                    # -------------------
                    # fig = utils.plot_images(image_batch[:16, ...], 
                    #                         dim=self.data.h_dim, 
                    #                         labels=decoded[:16, ...])
                    # plt.savefig('out/x_{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
                    # plt.close(fig)

                cnt += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.6f, g_loss: %.6f" \
                    % (epoch+1, self.config.epoch, batch+1, self.data.n_batches,
                    time.time() - start_time, self.err_D_fake+self.err_D_real, self.err_G))

                # record chkpt
                if np.mod(cnt, 500) == 2:
                    self.saver.save(self.sess,
                        os.path.join(self.config.chkp_dir, 'StratGAN'),
                        global_step=cnt)

    def sampler(self, z, image_batch, _labels=None, time=None):
        
        epoch = time[0]
        batch = time[1]

        samples, decoded = self.sess.run([self.G, self.decoder], 
                                          feed_dict={self.z: z, 
                                                     self.x: image_batch,    
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
