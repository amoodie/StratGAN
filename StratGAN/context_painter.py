import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from random import randint
import tensorflow as tf
import os


class ContextPainter(object):
    def __init__(self, stratgan,
                 paint_label=None, paint_width=1000, paint_height=None, 
                 paint_overlap=24, paint_overlap_thresh=10.0, 
                 paint_core_source='block',
                 paint_ncores=0, paint_core_thresh=0.01, 
                 batch_dim=40):

        print(" [*] Building painter...")

        self.sess = stratgan.sess
        self.stratgan = stratgan
        self.config = stratgan.config

        self.paint_samp_dir = self.stratgan.paint_samp_dir
        self.out_data_dir = self.stratgan.out_data_dir

        self.batch_dim = batch_dim

        if not paint_label == 0 and not paint_label:
            print('Label not given for painting, assuming zero for label')
            self.paint_label = np.zeros((batch_dim, stratgan.data.n_categories))
            self.paint_label[:, 0] = 1
            self.paint_int = 0
        else:
            # paint_label = tf.one_hot(paint_label, self.config.n_categories)
            self.paint_label = np.zeros((batch_dim, stratgan.data.n_categories))
            self.paint_label[:, paint_label] = 1
            self.paint_int = paint_label

        self.paint_width = paint_width
        if not paint_height:
            self.paint_height = int(paint_width / 4)
        else:
            self.paint_height = paint_height

        self.overlap = paint_overlap
        self.overlap_threshold = paint_overlap_thresh

        self.patch_height = self.patch_width = self.config.h_dim
        self.patch_size = self.patch_height * self.patch_width

        # self.canvas = np.ones((self.paint_height, self.paint_width))
        
        # generate the list of patch coordinates
        # self.patch_xcoords, self.patch_ycoords = self.calculate_patch_coords()
        # self.patch_count = self.patch_xcoords.size

        graph = tf.get_default_graph()
        self.gi = graph.get_tensor_by_name('gener/g_in:0')
        self.go = graph.get_tensor_by_name('gener/g_prob:0')
        self.do = graph.get_tensor_by_name('discr_1/Sigmoid:0')
        # self.gl = graph.get_tensor_by_name('loss_g_op:0')
        # self.gl = tf.log(1 - self.stratgan.D_fake)
        self.gl = tf.log(1 - self.do)
        # self.di = self.graph.get_tensor_by_name(model_name+'/'+disc_input)
        

        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        # load the last core arrays
        self.core_width = np.int(10)
        self.core_val = np.load(os.path.join(self.out_data_dir, 'last_core_val.npy'))
        self.core_loc = np.load(os.path.join(self.out_data_dir, 'last_core_loc.npy'))

        # self.mask0 = np.zeros((self.patch_width, self.patch_height))
        # self.image0 = np.zeros((self.patch_width, self.patch_height))
        # for i in np.arange(len(self.core_val)):
        #     self.mask0[:, self.core_loc[i]:self.core_loc[i]+self.core_width] = 1
        #     self.image0[:, self.core_loc[i]:self.core_loc[i]+self.core_width] = self.core_val[:,:,i]
        # self.x = tf.placeholder(tf.float32,
        #             [None, self.data.h_dim, self.data.w_dim, self.data.c_dim],
        #             name='x')
        # self.y = tf.placeholder(tf.float32, 
        #             [None, self.y_dim], 
                    # name='y') # labels


        # self.G_context = self.generator(_z=self.z_star, 
        #                                 _labels=self.y,
        #                                 is_training=False,
        #                                 batch_norm=self.stratgan.config.batch_norm,
        #                                 scope_name='gener_context')

        # self.mask0 = np.zeros((self.patch_width, self.patch_height), dtype=np.float32)
        # self.mask0[40:58,30:41] = 1
        # self.mask0[96:104,12:22] = 1
        # self.mask0[13:26,95:115] = 1
        # self.mask0 = self.mask0.flatten()
        # # self.mask = tf.convert_to_tensor(self.mask0, dtype=tf.float32)

        # self.image0 = 0.5 * np.ones((self.patch_width, self.patch_height), dtype=np.float32)
        # self.image0[40:58,30:35] = 0
        # self.image0[96:104,12:22] = 1
        # self.image0[13:26,95:115] = 0
        # self.image0 = self.image0.flatten()
        # print("image0: ", self.image0)
        # # self.image = tf.convert_to_tensor(self.image0, dtype=tf.float32)

        # self.patch0 = np.zeros((self.patch_width, self.patch_height), dtype=np.float32)

        ## MAKE THE MASK AND IMAGE
        self.z_0 = np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(np.float32)
        patch = self.sess.run(self.stratgan.G, 
                                  feed_dict={self.stratgan.z: self.z_0, 
                                             self.stratgan.y: self.paint_label[0,:].reshape(-1,self.stratgan.config.n_categories),
                                             self.stratgan.is_training: False})
        self.patch0 = patch.squeeze()

        randx = np.random.randint(low=0, high=self.patch_width, size=1000)
        randy = np.random.randint(low=0, high=self.patch_height, size=1000)

        self.mask0 = np.zeros((self.batch_dim, self.patch_width, self.patch_height), dtype=np.float32)
        self.mask0[:, randx, randy] = 1
        self.mask0 = self.mask0.reshape(self.batch_dim, -1)
        # print(self.mask0.shape)

        self.image0 = 0.5 * np.ones((self.batch_dim, self.patch_width, self.patch_height), dtype=np.float32)
        self.image0[:, randx, randy] = self.patch0[randx, randy]
        self.image0 = self.image0.reshape(self.batch_dim, -1)

        self.build_input_placeholders()
        self.build_context_loss()
        self.lam = 10.

        self.perceptual_loss = self.gl
        self.inpaint_loss = self.context_loss + self.lam*self.perceptual_loss
        self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)


    def build_context_loss(self):
        """Builds the context and prior loss objective"""
        # with self.graph.as_default():
        self.go = tf.reshape(self.go, [self.batch_dim, -1])
        self.context_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.masks, self.go) -
                       tf.multiply(self.masks, self.images))), 1)


    def build_input_placeholders(self):
      # with self.graph.as_default():
        self.masks = tf.placeholder(tf.float32,
                                    (self.batch_dim, self.patch_height*self.patch_width),
                                    name='masks')
        self.images = tf.placeholder(tf.float32,
                                     (self.batch_dim, self.patch_height*self.patch_width),
                                     name='images')
        # self.z_in = tf.placeholder(tf.float32,
        #                              self.stratgan.z_dim,
        #                              name='z_optim')


    def context_paint_image(self):

        self.mask_as_image = np.reshape(self.mask0[0,:], 
                                (self.patch_width, self.patch_height))
        self.image_as_image = np.reshape(self.image0[0,:], 
                                (self.patch_width, self.patch_height))
        self.patch0_as_image = np.reshape(self.patch0, 
                                (self.patch_width, self.patch_height))

        v = 0
        # self.z_inold = np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(np.float32)
        # print(self.z_inold.shape)
        
        momentum = 0.8
        lr = 0.001

        self.z_in = np.random.normal(-1, 1, [self.batch_dim, self.config.z_dim]).astype(np.float32)
        # self.z_in = tf.get_variable("z_in", [1, 100], tf.float32,
        #                     initializer=tf.random_uniform_initializer())

        self.writer = tf.summary.FileWriter(self.stratgan.train_log_dir,
                                            graph=self.sess.graph)
        self.writer.flush()

        # self.z_optim = tf.train.AdamOptimizer(lr, beta1=0.6) \
        #                  .minimize(self.inpaint_loss, var_list=self.z_in)

        # self.saver.save(self.sess,
        #                 os.path.join(self.train_chkp_dir, 'StratGAN'),
        #                 global_step=3000)

        # print("images: ", self.images)
        for i in np.arange(200):
            # out_vars = [self.stratgan.G, self.inpaint_loss, self.inpaint_grad]
            in_dict={self.stratgan.z: self.z_in, 
                     self.stratgan.y: self.paint_label,
                     self.stratgan.is_training: False,
                     self.masks: self.mask0,
                     self.images: self.image0}
          

            # patch, loss, grad = self.sess.run(self.G, z_Adam, feed_dict=in_dict)
            # patch, _ = self.sess.run([self.stratgan.G, self.z_optim], feed_dict=in_dict)
            out_vars = [self.inpaint_loss, self.inpaint_grad, self.go]
            loss, grad, patch = self.sess.run(out_vars, feed_dict=in_dict)

            # print("grad:", grad[0].shape)
            # print("loss:", loss)
            # print("v:", v)
                       
            # self.sess.run(tf.clip_by_value(self.z_in, -1, 1))

            # print("patch_shape:", patch.shape)
            # print("loss_shape:", loss.shape)
            # print("grad_shape:", grad.shape)
 
            if False:
                patch_reshaped = np.reshape(patch, (self.batch_dim, \
                                                self.patch_width, self.patch_height))
                fig = plt.figure()
                ax1 = fig.add_subplot(2,2,1)
                msk = ax1.imshow(self.mask_as_image, cmap='gray')
                msk.set_clim(0.0, 1.0)
                ax2 = fig.add_subplot(2,2,2)
                img = ax2.imshow(self.image_as_image, cmap='gray')
                img.set_clim(0.0, 1.0)
                ax3 = fig.add_subplot(2,2,3)
                ptch0 = ax3.imshow(self.patch0_as_image, cmap='gray')
                ptch0.set_clim(0.0, 1.0)
                ax4 = fig.add_subplot(2,2,4)
                best_patch_idx = np.argmin(np.sum(loss,1),0)
                ptch = ax4.imshow(patch_reshaped[best_patch_idx,:,:], cmap='gray')
                ptch.set_clim(0.0, 1.0)
                # plt.savefig(os.path.join(self.paint_samp_dir, 'context_{}.png'.format(str(i).zfill(3))), 
                plt.savefig(os.path.join(self.paint_samp_dir, 'context_i.png'), 
                            bbox_inches='tight', dpi=150)
                plt.close()

            v_prev = np.copy(v)
            v = momentum*v - lr*grad[0]
            self.z_in += (-momentum * v_prev +
                     (1 + momentum) * v)
            self.z_in = np.clip(self.z_in, -1, 1)

            # print("z shape:", z.shape)
            verbose = True
            if verbose:
                print('Iteration {}: {}'.format(i, np.mean(loss)))
                # print('z_in: {}'.format(self.z_in[0, 0:5]))
                # print("perceptual loss:", self.perceptual_loss)
            
        self.patchF = np.copy(patch)
        # return patchF



        # z = np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(np.float32)
        # paint_label = self.paint_label
        # patch = self.sess.run(self.stratgan.G, feed_dict={self.stratgan.z: z, 
        #                                      self.stratgan.y: paint_label,
        #                                      self.stratgan.is_training: False})
        # r_patch = patch[0].reshape(self.config.h_dim, self.config.h_dim)
        # # return r_patch

        # print("perceptual loss:", self.perceptual_loss)