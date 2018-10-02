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

        
        # instantiate placeholders:
        # -------------------
        self.x = tf.placeholder(tf.float32,
                    [self.data.batch_size, self.data.h_dim * self.data.w_dim], name='x')
        self.y = tf.placeholder(tf.float32, 
                    [self.data.batch_size, self.y_dim], 
                    name='y') # labels
        self.z = tf.placeholder(tf.float32, 
                    shape=[self.data.batch_size, self.config.z_dim], 
                    name='z') # generator inputs
        self.summ_z = tf.summary.histogram('z', self.z)


        # instantiate networks:
        # -------------------
        self.G                          = self.generator(self.z, 
                                                         self.y,
                                                         batch_norm=self.config.batch_norm)
        self.D_real, self.D_real_logits = self.discriminator(self.x, 
                                                             self.y, 
                                                             reuse=False,
                                                             batch_norm=self.config.batch_norm) # real response
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, 
                                                             self.y, 
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

        self.summ_loss_d_real = tf.summary.scalar("loss_d", self.loss_d_real)
        self.summ_loss_d_fake = tf.summary.scalar("loss_d_", self.loss_d_fake)

        self.summ_loss_g = tf.summary.scalar("loss_g", self.loss_g)
        self.summ_loss_d = tf.summary.scalar("loss_d", self.loss_d)

        self.summ_image = tf.summary.histogram("images", self.x)
        self.summ_label = tf.summary.histogram("labels", self.y)
        self.summ_z     = tf.summary.histogram("zs", self.z)
        self.summ_input = tf.summary.merge([self.summ_image, self.summ_label, self.summ_z])


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def generator(self, z, _labels=None, batch_norm=False):
        
        _out_size = self.data.w_dim * self.data.h_dim

        with tf.variable_scope('gener') as _scope:
        
            g_c1 = tf.concat([z, _labels], axis=1, name='g_c1')
            g_h1 = ops.leaky_relu_layer(g_c1, _out_size // 4,
                                        scope='g_h1', batch_norm=batch_norm)
            g_c2 = tf.concat([g_h1, _labels], axis=1, name='g_c2')
            g_h2 = ops.leaky_relu_layer(g_c2, _out_size // 2,
                                        scope='g_h2', batch_norm=batch_norm)
            g_c3 = tf.concat([g_h2, _labels], axis=1, name='g_c3')
            g_prob = ops.sigmoid_layer(g_c3, _out_size, scope='g_prob')

            return g_prob


    def discriminator(self, _images, _labels=None, reuse=False, batch_norm=False):
        
        _in_size = self.data.w_dim * self.data.h_dim

        with tf.variable_scope('discr') as scope:
            if reuse:
                scope.reuse_variables()

            d_c1 = tf.concat([_images, _labels], axis=1, name='d_c1')
            d_h1 = ops.leaky_relu_layer(d_c1, _in_size // 2, 
                                        scope='d_h1', batch_norm=batch_norm)

            d_c2 = tf.concat([d_h1, _labels], axis=1, name='d_c2')
            d_h2 = ops.leaky_relu_layer(d_c2, _in_size // 4, 
                                        scope='d_h2', batch_norm=batch_norm)

            # d_c3 = tf.concat([d_h2, _labels], axis=1, name='d_c3')
            # d_h3 = ops.linear_layer(d_c3, 1, scope='d_prob')

            d_c4 = tf.concat([d_h2, _labels], axis=1, name='d_c4')

            return tf.nn.sigmoid(d_c4), d_c4


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
        self.summ_d = tf.summary.merge([self.summ_z, self.summ_D_real, 
                                        self.summ_loss_d_real, self.summ_loss_d])
        self.writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

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
                
                # label_batch = _label_batch.eval()
                label_batch = _label_batch.copy()


                self.decoder = tf.argmax(label_batch, axis=1)


                # update networks:
                # -------------------
                # update D network
                _, summary_str = self.sess.run([d_optim, self.summ_d], 
                                                feed_dict={self.x: image_batch,
                                                           self.y: label_batch,
                                                           self.z: z_batch})
                self.writer.add_summary(summary_str, cnt)
                
                # update G network
                _, summary_str = self.sess.run([g_optim, self.summ_g], 
                                                feed_dict={self.z: z_batch,
                                                           self.y: label_batch})
                self.writer.add_summary(summary_str, cnt)
                
                # update G network
                _, summary_str = self.sess.run([g_optim, self.summ_g], 
                                                feed_dict={self.z: z_batch,
                                                           self.y: label_batch})
                self.writer.add_summary(summary_str, cnt)


                self.err_D_fake = self.loss_d_fake.eval({ self.z: z_batch, self.y: label_batch })
                self.err_D_real = self.loss_d_real.eval({ self.x: image_batch, self.y: label_batch })
                self.err_G      = self.loss_g.eval({ self.z: z_batch, self.y: label_batch })


                if batch % 10 == 0:
                    gen_samples, decoded = self.sess.run([self.G, self.decoder], 
                                                          feed_dict={self.z: z_batch, 
                                                                     self.y: label_batch})
                    fig = utils.plot_images(gen_samples[:16, ...], 
                                            dim=self.data.h_dim, 
                                            labels=decoded[:16, ...])
                    plt.savefig('samp/g_{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

                    # make plot of input images:
                    # -------------------
                    fig = utils.plot_images(image_batch[:16, ...], 
                                            dim=self.data.h_dim, 
                                            labels=decoded[:16, ...])
                    plt.savefig('out/x_{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

                cnt += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.6f, g_loss: %.6f" \
                    % (epoch+1, self.config.epoch, batch+1, self.data.n_batches,
                    time.time() - start_time, self.err_D_fake+self.err_D_real, self.err_G))

                # record chkpt
                if np.mod(cnt, 500) == 2:
                    self.saver.save(self.sess,
                        os.path.join(self.config.chkp_dir, 'StratGAN'),
                        global_step=cnt)
