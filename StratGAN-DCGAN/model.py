import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys

import loader
import ops
import utils
import painter

# from pympler.tracker import SummaryTracker, summary, muppy
# tracker = SummaryTracker()
# import types
# from pympler import asizeof

# import gc


class StratGAN(object):
    def __init__(self, sess, config): 
        
        print('\n [*] Initializing model...')
        self.sess = sess
        self.config = config

        # Load the dataset
        print(' [*] Building dataset provider...')
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

        # grab some info from the data into the config
        self.config.h_dim = self.data.h_dim
        self.config.w_dim = self.data.w_dim
        self.config.n_categories = self.data.n_categories

        # Initialize the net model
        print(' [*] Building model...')
        self.build_model()

        utils.write_config(self)


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
                          name='encoded') # generator label inputs
        self.is_training = tf.placeholder(tf.bool, name='is_training')


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
                                                             batch_norm=self.config.batch_norm,
                                                             minibatch=self.config.minibatch_discrim) # real response
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, 
                                                             self.y, 
                                                             reuse=True,
                                                             is_training=self.is_training,
                                                             batch_norm=self.config.batch_norm,
                                                             minibatch=self.config.minibatch_discrim) # fake response

        # decoder to convert one-hot labels to category numbers
        self.decoder = tf.argmax(self.encoded, axis=1)


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


        # define summary stats
        # -------------------
        self.summ_D_real = tf.summary.histogram("D_real", self.D_real)
        self.summ_D_fake = tf.summary.histogram("D_fake", self.D_fake)
        self.summ_G = tf.summary.image("G", tf.reshape(self.G, 
                                       [self.config.batch_size, self.data.h_dim, self.data.w_dim, -1]))

        self.summ_loss_g = tf.summary.scalar("loss_g", self.loss_g)
        self.summ_loss_d = tf.summary.scalar("loss_d", self.loss_d)

        self.summ_loss_d_real = tf.summary.scalar("loss_d_real", self.loss_d_real)
        self.summ_loss_d_fake = tf.summary.scalar("loss_d_fake", self.loss_d_fake)

        self.summ_image = tf.summary.histogram("images", self.x)
        self.summ_label = tf.summary.histogram("labels", self.y)
        self.summ_z     = tf.summary.histogram("zs", self.z)


        # setup trainable
        # -------------------
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]


        # a few more initializations
        # -------------------
        # directories for logging the training 
        self.train_log_dir = os.path.join(self.config.log_dir, self.config.run_dir)
        self.train_samp_dir = os.path.join(self.config.samp_dir, self.config.run_dir)
        self.train_chkp_dir = os.path.join(self.config.chkp_dir, self.config.run_dir)

        self.saver = tf.train.Saver()


    def generator(self, _z, _labels, is_training, batch_norm=False):
        
        print(' [*] Building generator...')

        _batch_size = tf.shape(_z)[0] # dynamic batch size op
        
        with tf.control_dependencies([_batch_size]):

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


    def discriminator(self, _images, _labels, is_training,
                      reuse=False, batch_norm=False, minibatch=False):
        
        print(' [*] Building discriminator...')

        flat_shape = int( (self.data.w_dim / 4)**2 * (self.config.df_dim + self.data.n_categories) )

        with tf.variable_scope('discr') as scope:
            if reuse:
                scope.reuse_variables()

            # reshape the labels for concatenation to feature axis of conv tensors
            _labels_r = tf.reshape(_labels, [-1, 1, 1, self.y_dim])

            # convolution, layer 0
            d_c0 = ops.condition_conv_concat([_images, _labels_r], axis=3, 
                                             name='d_cat0')
            d_h0 = ops.conv2d_layer(d_c0, self.data.c_dim + self.data.y_dim, 
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

            # fully connected, layer 2
            d_r2 = tf.reshape(d_h1, [-1, flat_shape])
            d_c2 = ops.condition_concat([d_r2, _labels], axis=1, 
                                        name='d_cat2')
            
            d_h2 = ops.linear_layer(d_c2, self.config.dfc_dim, 
                                    is_training=is_training, 
                                    scope='d_h2', batch_norm=batch_norm)
            d_h2 = tf.nn.leaky_relu(d_h2, alpha=self.config.alpha)

            # minibatch discrim, optional layer
            if minibatch:
                d_h2 = ops.minibatch_discriminator_layer(d_h2, num_kernels=5, kernel_dim=3)

            # fully connected, layer 3
            d_c3 = ops.condition_concat([d_h2, _labels], axis=1, 
                                        name='d_cat3')
            d_h3 = ops.linear_layer(d_c3, 1, 
                                    is_training=False, 
                                    scope='d_h3', batch_norm=False)
            d_prob = tf.nn.sigmoid(d_h3)

            return d_prob, d_h3


    def train(self):

        print(' [*] Beginning training...')
        # solvers:
        # -------------------
        d_optim = tf.train.AdamOptimizer(self.config.learning_rate, 
                                         beta1=self.config.beta1) \
                                .minimize(self.loss_d, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.config.learning_rate, 
                                         beta1=self.config.beta1) \
                                .minimize(self.loss_g, var_list=self.g_vars)
        
        # initialize all variables
        z_batch = np.zeros(([self.config.batch_size, self.config.z_dim]))
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.z: z_batch})

        # initialize summary variables recorded during training
        self.summ_g = tf.summary.merge([self.summ_D_fake, self.summ_G, 
                                        self.summ_loss_d_fake, self.summ_loss_g])
        self.summ_d = tf.summary.merge([self.summ_D_real, self.summ_loss_d_real, 
                                        self.summ_loss_d])
        self.summ_input = tf.summary.merge([self.summ_image, self.summ_label, 
                                            self.summ_z])
        self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)

        # set of training random z and label tensors for training gifs
        self.training_zs, self.training_labels = utils.training_sample_set(
                                                        self.config.z_dim, 
                                                        self.data.n_categories)

        cnt = 0
        start_time = time.time()
        print("Start time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

        # finalize to make sure no more ops are added!
        self.sess.graph.finalize()

        for epoch in np.arange(self.config.epoch):

            for batch in np.arange(self.data.n_batches):
                

                # grab the new batch:
                # -------------------
                _image_batch, _label_batch = self.sess.run(self.data.next_batch)

                z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]) \
                                            .astype(np.float32)

                # copy to prevent consumption during training
                image_batch = _image_batch.copy()
                label_batch = _label_batch.copy()

                # optional augmentation
                if self.config.noisy_inputs:
                    image_batch = image_batch + 1 * np.random.normal(0, 0.1, size=image_batch.shape)
                if self.config.flip_inputs:
                    image_batch = 1 - image_batch


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
                                                               self.y: label_batch,
                                                               self.is_training: True})
                self.writer.add_summary(summary_str, cnt)

                # calculate new errors for printing
                self.err_D_fake = self.loss_d_fake.eval({ self.z: z_batch, 
                                                          self.y: label_batch,
                                                          self.is_training: False })
                self.err_D_real = self.loss_d_real.eval({ self.x: image_batch, 
                                                          self.y: label_batch,
                                                          self.is_training: False })
                self.err_G      = self.loss_g.eval({ self.z: z_batch, 
                                                     self.y: label_batch,
                                                     self.is_training: False })

                # make records and samples:
                # -------------------
                # sample interval
                if cnt % 20 == 0:
                    self.sampler(self.training_zs, _labels=self.training_labels, 
                                 train_time=[epoch, batch], samp_dir=self.train_samp_dir)

                # record chkpt
                if np.mod(cnt, 500) == 2:
                    self.saver.save(self.sess,
                                    os.path.join(self.train_chkp_dir, 'StratGAN'),
                                    global_step=cnt)

                # print the current training state
                cnt += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.6f, g_loss: %.6f" \
                    % (epoch+1, self.config.epoch, batch+1, self.data.n_batches,
                    time.time() - start_time, self.err_D_fake+self.err_D_real, self.err_G))

                
                # debugging memory leaking:
                # -------------------
                # objList = muppy.get_objects()
                # my_types = muppy.filter(objList, Type=(list))
                # sum1 = summary.summarize(objList)
                # summary.print_(sum1)

                # loadersize = utils.getsize(self.data)
                # print('loadersize:', loadersize)

                # loadersize = asizeof.asizeof(self.data)
                # print('loadersize:', loadersize)

                # loadersize = asizeof.asizeof(self)
                # print('self', loadersize)

                # for obj in gc.get_objects():
                #     if isinstance(obj, list):
                #         print(obj)


    def sampler(self, z, _labels=None, train_time=None, samp_dir='samp'):
        
        if not train_time:
            train_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            samp_name = 'g_{0}.png'.format(train_time)
        else:
            epoch = train_time[0]
            batch = train_time[1]
            samp_name = 'g_{0}_{1}.png'.format(str(epoch+1).zfill(3), 
                                                str(batch).zfill(4))

        samples, decoded = self.sess.run([self.G, self.decoder], 
                                          feed_dict={self.z: z, 
                                                     self.y: _labels,
                                                     self.encoded: _labels,
                                                     self.is_training: False})
        fig = utils.plot_images(samples, image_dim=self.data.h_dim, 
                                n_categories=self.data.n_categories, 
                                labels=decoded)

        file_name = os.path.join(samp_dir, samp_name)

        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)
        print("Sample: {file_name}".format(file_name=file_name))


    def load(self, checkpoint_dir):
            import re
            print(" [*] Reading checkpoints...")

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0


    def paint(self):

        # paint_label = self.config.paint_label
        # paint_height = self.config.paint_height
        # paint_width = self.config.paint_width
        # patch_height = patch_width = self.data.h_dim

        # directories for logging the painting
        self.paint_samp_dir = os.path.join(self.config.paint_dir, self.config.run_dir)

        # initialize the painter object
        self.painter = painter.CanvasPainter(self, paint_label=self.config.paint_label, 
                                                   paint_width=self.config.paint_width)

        # sample now initialized
        samp = plt.imshow(self.painter.canvas, cmap='gray')
        plt.plot(self.painter.patch_xcoords, self.painter.patch_ycoords, marker='.', ls='none')
        plt.savefig(os.path.join(self.paint_samp_dir, 'init.png'), bbox_inches='tight')
        plt.close()

        self.painter.fill_canvas()

        samp = plt.imshow(self.painter.canvas, cmap='gray')
        # plt.plot(self.painter.patch_xcoords, self.painter.patch_ycoords, marker='o', ls='none')
        plt.savefig(os.path.join(self.paint_samp_dir, 'final.png'), bbox_inches='tight', dpi=300)
        plt.close()

