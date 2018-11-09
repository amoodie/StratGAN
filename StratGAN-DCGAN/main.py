import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils import Config, mkdirs, rand_id
from model import StratGAN  

"""
### MINIREADME ###

options for directories:
['multi_line','multi_line_bw','shapes_all','shapes_all_mini','shapes_star','shapes_circle']
must be specified as below ../data/


"""


# Setup configuration
# -----------
flags = tf.app.flags

# training related flags
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("epoch", 5, "Epoch to train [5]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.6, "Momentum term of adam [0.6]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("image_dir", "multi_line", "Root directory of dataset [multi_line]")
flags.DEFINE_string("run_dir", None, "Directory run name to save samp, log, chkp under. If none, auto select [None]")
# flags.DEFINE_integer("sample_int", 100, "The interval to sample images at during training [100]")

# painting related flags
flags.DEFINE_boolean("paint", False, "True for painting, False for painting [False]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("output_width", 2000, "The size of the output images to produce. If None, same value as output_height [2000]")
flags.DEFINE_integer("output_height", None, "The size of the output images to produce. If None, value of output_width/4 [None]")

# create flag object
FLAGS = flags.FLAGS

# merge flags and fixed configs into config, which gets passed to the StratGAN object
config = Config()
config.image_dir = os.path.join(os.pardir, 'data', FLAGS.image_dir)
config.image_dir = os.path.join(os.pardir, 'data', 'shapes_all')
config.image_ext = '*.png'
config.img_verbose = True
config.c_dim = 1

config.batch_size = FLAGS.batch_size
config.repeat_data = True
config.shuffle_data = True
config.buffer_size = 4
config.drop_remainder = True            # currently fails if false!

config.z_dim = 100                      # number inputs to gener
config.gf_dim = 64                      # number of gener conv filters
config.df_dim = 64                      # number of discim conv filters
config.gfc_dim = 1024                   # number of gener fully connecter layer units
config.dfc_dim = 1024                   # number of discim fully connected layer units

config.alpha = 0.1                      # leaky relu alpha

config.epoch = FLAGS.epoch
config.learning_rate = FLAGS.learning_rate          # optim learn rate
config.beta1 = FLAGS.beta1                      # momentum
config.batch_norm = True
config.minibatch_discrim = True
config.gener_iter = 4                   # times to update generator per discriminator update
config.noisy_inputs = False             # add some small noise to the input images
config.flip_inputs = False              # whether to flip the black white pixels

config.log_dir = 'log'
config.out_dir = 'out'
config.samp_dir = 'samp'
config.chkp_dir = 'chkp'
config.paint_dir = 'paint'
config.run_dir = FLAGS.run_dir
if not config.run_dir: # if the run dir was not given, make something up
    config.run_dir = rand_id()


# create folder structure
# -----------
folder_list = [config.out_dir, config.log_dir, config.samp_dir]
mkdirs(folder_list)
mkdirs([os.path.join(config.log_dir, config.run_dir), 
        os.path.join(config.samp_dir, config.run_dir)])


# what to do with the model?
# -----------



# model execution function
# -----------
def main(_):
    with tf.Session() as sess:

        # Initiate session and initialize all vaiables
        stratgan = StratGAN(sess, config)

        if FLAGS.train:
            stratgan.train()

        elif FLAGS.paint:
            paint_chkp_dir = os.path.join(config.chkp_dir, config.run_dir)
            stratgan.load(paint_chkp_dir)
            stratgan.paint()

        else:
            print('Neither "train" nor "paint" selected. Doing nothing.')

if __name__ == '__main__':
    tf.app.run()
