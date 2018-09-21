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

        # Initialize the net model
        self.build_model()


    def build_model(self):

        self.y_dim = self.config.y_dim
        self.z_dim = self.config.z_dim

        # labels
        self.y = tf.placeholder(tf.float32, [self.config.batch_size, self.y_dim], name='y')

        # generator inputs
        self.z = tf.placeholder(tf.float32, shape=[None, self.config.z_dim], name='z')
        self.summ_z = tf.summary.histogram('z', self.z)



        # HAVENT TOUCHED THIS YET!!
        # X = tf.placeholder(tf.float32, shape=[None, 784])

        # D_W1 = tf.Variable(xavier_init([784, 128]))
        # D_b1 = tf.Variable(tf.zeros(shape=[128]))

        # D_W2 = tf.Variable(xavier_init([128, 1]))
        # D_b2 = tf.Variable(tf.zeros(shape=[1]))

        # theta_D = [D_W1, D_W2, D_b1, D_b2]

        # G_W1 = tf.Variable(xavier_init([100, 128]))
        # G_b1 = tf.Variable(tf.zeros(shape=[128]))

        # G_W2 = tf.Variable(xavier_init([128, 784]))
        # G_b2 = tf.Variable(tf.zeros(shape=[784]))

        # theta_G = [G_W1, G_W2, G_b1, G_b2]




        # instantiate networks:
        # -------------------
        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(self.data.image_batch, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D__logits = self.discriminator(self.G, self.y, reuse=True)

        # alternative losses:
        # -------------------
        # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        # G_loss = -tf.reduce_mean(tf.log(D_fake))
        
        # why ones like?
        self.loss_d_real = tf.reduce_mean(ops.scewl(logits=self.D_logits, 
                                                    labels=tf.ones_like(self.D)))
        self.loss_d_fake = tf.reduce_mean(ops.scewl(logits=self.D__logits, 
                                                    labels=tf.zeros_like(self.D_)))
        self.loss_g = tf.reduce_mean(ops.scewl(logits=self.D__logits, 
                                               labels=tf.ones_like(self.D_)))

        self.summ_loss_d = tf.summary.scalar("loss_d", self.loss_d)
        self.summ_loss_d_ = tf.summary.scalar("loss_d_", self.loss_d_)

        self.loss_d = self.loss_d_real + self.loss_d_loss

        self.summ_loss_g = tf.summary.scalar("loss_g", self.loss_g)
        self.summ_loss_d = tf.summary.scalar("loss_d", self.loss_d)

        # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        # D_loss = D_loss_real + D_loss_fake
        # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


        


        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def generator(self, z, y=None):
        with tf.variable_scope("gener") as scope:
            g_h1 = ops.relu_layer(z, 128, scope='g_h1')
            g_h2 = ops.relu_layer(g_h1, 512, scope='g_h2')
            g_prob = ops.sigmoid_layer(g_h2, 784, scope='g_prob')

            ### OLD VERSION:
            # G_W1 = tf.Variable(ops.xavier_init([self.z_dim, 128]))
            # G_b1 = tf.Variable(tf.zeros(shape=[128]))
            # G_W2 = tf.Variable(ops.xavier_init([128, 784]))
            # G_b2 = tf.Variable(tf.zeros(shape=[784]))
            # G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            # G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            # G_prob = tf.nn.sigmoid(G_log_prob)

            return g_prob


    def discriminator(self, image, label=None, reuse=False):
        with tf.variable_scope("discr") as scope:
            if reuse:
                scope.reuse_variables()

            d_h1 = ops.relu_layer(image, 512, scope='d_h1')
            d_h2 = ops.relu_layer(d_h1, 128, scope='d_h2')
            d_prob, d_logits, _ = ops.sigmoid_layer(d_h2, 1, scope='d_prob', return_w=True)

            ### OLD VERSION:
            # D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
            # D_logit = tf.matmul(D_h1, D_W2) + D_b2
            # D_prob = tf.nn.sigmoid(D_logit)

            return d_prob, d_logits

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