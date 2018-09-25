import tensorflow as tf
import numpy as np

import loader
import ops


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

        # could grab the data here if needed??
        # self.image_batch, self.label_batch = self.data.iterator.get_next()
        # self.sess.run(self.image_batch, self.label_batch)

        # print(self.image_batch)
        # grab some info from the data provider
        # self.batch_size = tf.shape(self.data.image_batch)
        self.batch_size = self.config.batch_size
        # print("config batch size:", self.batch_size)

        # Initialize the net model
        self.build_model()


    def build_model(self):

        # grab some parameters for convenience:
        # -------------------
        self.y_dim = self.config.y_dim
        self.z_dim = self.config.z_dim

        
        # instantiate placeholders:
        # -------------------
        self.y = tf.placeholder(tf.float32, 
                    [self.config.batch_size, self.y_dim], 
                    name='y') # labels
        self.z = tf.placeholder(tf.float32, 
                    shape=[self.config.batch_size, self.config.z_dim], 
                    name='z') # generator inputs
        self.summ_z = tf.summary.histogram('z', self.z)


        # instantiate networks:
        # -------------------
        self.G                          = self.generator(self.z, 
                                                         self.y)
        self.D_real, self.D_real_logits = self.discriminator(self.data.image_batch, 
                                                             self.y, 
                                                             reuse=False) # real response
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, 
                                                             self.y, 
                                                             reuse=True) # fake response
        # self.sampler = self.sampler(self.z, self.y)


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

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def generator(self, z, y=None):
        with tf.variable_scope('gener') as _scope:
            g_h1 = ops.relu_layer(z, 256, scope='g_h1')
            g_h2 = ops.relu_layer(g_h1, 1024, scope='g_h2')
            g_prob = ops.sigmoid_layer(g_h2, 4096, scope='g_prob')

            return g_prob


    def discriminator(self, _images, label=None, reuse=False):
        with tf.variable_scope('discr') as scope:
            if reuse:
                scope.reuse_variables()

            _images = tf.reshape(_images, [self.batch_size, 
                                           self.data.h_dim * self.data.w_dim])

            d_h1 = ops.relu_layer(_images, 512, scope='d_h1')
            d_h2 = ops.relu_layer(d_h1, 128, scope='d_h2')
            d_h3 = ops.linear_layer(d_h2, 1, scope='d_prob')

            return tf.nn.sigmoid(d_h3), d_h3

    def train(self, config):

        # solver:
        # -------------------
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

        ### FROM DCGAN:
        # d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #       .minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #       .minimize(self.g_loss, var_list=self.g_vars)
        
        sess.run(tf.global_variables_initializer())

        # load up the dataset (all paths and labels?)

        cnt = 0
        start_time = time.time()
        for epoch in np.arange(config.epoch):
            
            # shuffle dataset

            for batch in np.arange(num_batches):
                
                # need to store the labels and images as fixed so they can be
                #   used multiple times by the discr and gener
                self. y = self.data.label_batch


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