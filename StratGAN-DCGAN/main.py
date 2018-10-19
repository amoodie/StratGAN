import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils import Config, mkdirs
from model import StratGAN  



# Setup configuration
# -----------
config = Config()
# config.image_dir = os.path.join(os.pardir, 'data', 'multi_line')
# config.image_dir = os.path.join(os.pardir, 'data', 'multi_line_bw')
# config.image_dir = os.path.join(os.pardir, 'data', 'shapes_all')
# config.image_dir = os.path.join(os.pardir, 'data', 'shapes_star')
config.image_dir = os.path.join(os.pardir, 'data', 'shapes_circle')
config.image_ext = '*.png'
config.img_verbose = True
config.c_dim = 1

config.batch_size = 100
config.repeat_data = False
config.shuffle_data = True
config.buffer_size = 4
config.drop_remainder = True            # currently fails if false!

config.z_dim = 100                      # number inputs to gener
config.gf_dim = 64                      # number of gener conv filters
config.df_dim = 64                      # number of discim conv filters
config.gfc_dim = 1024                   # number of gener fully connecter layer units
config.dfc_dim = 1024                   # number of discim fully connected layer units

config.alpha = 0.1                      # leaky relu alpha

config.epoch = 1
config.learning_rate = 0.002            # optim learn rate
config.beta1 = 0.6                      # momentum
config.batch_norm = False
config.gener_iter = 3                   # times to update generator per discriminator update
config.noisy_inputs = False              # add some small noise to the input images
config.flip_inputs = False              # whether to flip the black white pixels

config.log_dir = 'log'
config.out_dir = 'out'
config.samp_dir = 'samp'
config.chkp_dir = 'chkp'


# create folder structure
# -----------
mkdirs(config)


# model execution
# -----------
with tf.Session() as sess:
    
    # Initiate session and initialize all vaiables
    stratgan = StratGAN(sess, config)
    
    stratgan.train()
