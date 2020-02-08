import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import cv2
import tensorflow as tf 



class StratHeteroProvider(object):
    def __init__(self, batch_size=64):

        self.data_generator = StratHeteroGenerator(n_categories=None)
        self.data = tf.data.Dataset.from_generator(lambda: self.data_generator,
                                                     (tf.float32, tf.float32),
                                                     (tf.TensorShape((64, 64, 1)), 
                                                      tf.TensorShape((self.data_generator.n_categories))))
        
        # self.data = self.data.map(self._parse_function, num_parallel_calls=num_threads)
        # self.data = self.data.prefetch(prefetch_buffer)
        self.batch_size = batch_size
        self.data = self.data.batch(self.batch_size)

        # create iterator and final input tensors
        self.iterator = self.data.make_one_shot_iterator()
        
        # self.image_batch, self.label_batch = self.iterator.get_next()
        
        self.next_batch = self.iterator.get_next()

        # self.data = self.data.prefetch(1)

        self.data.h_dim = self.data.w_dim = 64
        self.h_dim, self.w_dim = self.data.h_dim, self.data.w_dim
        self.c_dim = self.data.c_dim = 1
        self.data.n_categories = self.n_categories = self.data_generator.n_categories 
        self.data.y_dim = self.y_dim = self.n_categories       
        self.data_shape = [self.batch_size, self.h_dim, self.w_dim, self.c_dim]

        self.n_batches = 50



class StratHeteroGenerator(object):
    def __init__(self, width=64, height=64, nc_min=1, nc_max=10,
                 cw_mu=30, cw_sig=10, ch_mu=6, ch_sig=1,
                 batch_size=64, n_categories=None):
        """
        Inputs:
            width : image width
            height : image height
            nc_min : min number of channels
            mc_max : 
            cw_mu30 :
            cw_sig :
            ch_mu : 
            ch_sig :
        """

        ## set up the variables
        # channel scale heterogeneity follows a discrete uniform distribution
        self.nc_min = nc_min # min number of channels
        self.nc_max = nc_max # max number of channels

        # bed scale heretogeneity follows a truncated normal distribution (0,1)
        self.P_b_mu = 0 # mean of draw
        self.P_b_sig = 0.25 # std of dist

        # channel size dists
        self.cw_mu = cw_mu # width mean
        self.ch_mu = ch_mu # height mean
        self.cw_sig = cw_sig # width std
        self.ch_sig = ch_sig# height std

        self.nx = width # cols
        self.ny = height # rows
        self.empty = np.zeros((self.nx, self.ny)) # init a strike section array
        # self.empty.fill(np.nan)

        # other info
        self.batch_size = batch_size
        if not n_categories:
            self.n_categories = self.nc_max
        else:
            self.n_categories = n_categories

        # list of distortion functions to pick from
        self.dist_func = [self.random_gaussian_blur,
                          self.random_brightness_contrast,
                          self.random_noise]

    def __iter__(self):
        return self

    def __next__(self):
        # output is a single np.array to train with
        strk = self.empty.copy()
        nc = np.random.randint(low=self.nc_min, high=self.nc_max+1) # number of channels

        # P_b_mu = 0 #np.random.normal(loc=0, scale=0.5) # mean of reduction from 1
        img_P_b_sig = np.abs(np.random.normal(loc=self.P_b_mu, scale=self.P_b_sig)) + 1e-6 # std of dist
        for c in np.arange(nc):
            cw = np.round(np.random.normal(loc=self.cw_mu, scale=self.cw_sig)).astype(np.int)
            ch = np.round(np.random.normal(loc=self.ch_mu, scale=self.ch_sig)).astype(np.int)
            cw, ch = np.clip(cw, 0, self.nx-1), np.clip(ch, 0, self.ny-1)

            cl_w = np.random.randint(low=0,high=self.nx-cw) # channel lower left x
            cl_h = np.random.randint(low=0,high=self.ny-ch) # channel lower left y

            c_het = np.abs( np.random.normal(loc=self.P_b_mu, scale=img_P_b_sig, size=(ch,cw)) ) # channel heterogeneity matrix
            c_het = np.clip(c_het, 0, 0.5) # clip to range

            strk[cl_h:cl_h+ch,cl_w:cl_w+cw] = 0.5 + c_het # replace values in strk with hetero matrix

        # apply distortion
        # if np.random.uniform() < 0.5:
        #     strk = self.random_gaussian_blur(strk)

        # make the label vector a one-hot
        label = nc-1
        a = np.array([label])
        b = np.zeros((self.n_categories))
        b[a] = 1
        # one_hot_label = tf.one_hot(label, self.n_categories)
        one_hot_label = b.astype(np.float32)

        return np.expand_dims(strk,2), one_hot_label


    def random_gaussian_blur(self, image):
        k_size = np.random.choice(np.arange(1, 5, 2))
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)
        image = np.minimum(np.maximum(image, 0), 1)
        return image

    def random_brightness_contrast(self, image):
        brightness = 0.25 + random.random() / 2
        contrast = 0.25 + random.random() / 2
        image = contrast * (image - np.mean(image)) / np.std(image) + brightness
        image = np.minimum(np.maximum(image, 0), 1)
        return image

    def random_noise(self, image):
        noise_var = random.random() / 20
        noise = np.random.randn(image.shape[0], image.shape[1]) * noise_var
        image += noise
        image = np.minimum(np.maximum(image, 0), 1)
        return image




if __name__ == '__main__':
    dg = StratHeteroProvider(batch_size=50)
    for i in np.arange(dg.batch_size):
        img = next(dg)
        cv2.imwrite("imgs/{0}.png".format(str(i).zfill(3)), img * 255)
        # cv2.imwrite("mask.png", (input_mask * 255).astype(np.uint8))
        # cv2.imwrite("width_map.png", width_map)


        # add it to the plot
        # ax = plt.subplot(gs[i])
        # strk_cnv = ax.imshow(strk, cmap='gray_r')
        # current_cmap = cm.get_cmap()
        # current_cmap.set_bad(color='white')
        # strk_cnv.set_clim(0, 1)
        # ax.set_xticks([])
        # ax.set_yticks([])