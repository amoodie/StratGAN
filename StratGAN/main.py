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

# general flags
flags.DEFINE_string("run_dir", None, "Directory run name to save/load samp, log, chkp under. If none, auto select [None]")
flags.DEFINE_integer("gf_dim", 64, "Number of filters in generator [64]")
flags.DEFINE_integer("df_dim", 64, "Number of filters in discriminator [64]")

# training related flags
flags.DEFINE_boolean("train", False, "True for training [False]")
flags.DEFINE_integer("epoch", 5, "Epoch to train [5]")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate of for adam [0.0005]")
flags.DEFINE_float("beta1", 0.6, "Momentum term of adam [0.6]")
flags.DEFINE_integer("batch_size", 64, "Size of batch images [64]")
flags.DEFINE_integer("gener_iter", 2, "Number of times to iterate generator per batch [2]")
flags.DEFINE_string("image_dir", "multi_line_bw_128", "Root directory of dataset [multi_line_bw_128]")
# flags.DEFINE_integer("sample_int", 100, "The interval to sample images at during training [100]")

# painting related flags
flags.DEFINE_boolean("paint", False, "True for painting [False]")
flags.DEFINE_integer("paint_label", None, "The label to paint with")
flags.DEFINE_integer("paint_width", 1000, "The size of the paint images to produce. If None, same value as paint_height [1000]")
flags.DEFINE_integer("paint_height", None, "The size of the paint images to produce. If None, value of paint_width/4 [None]")
flags.DEFINE_integer("paint_overlap", 24, "The size of the overlap during painting [24]")
flags.DEFINE_string("paint_patcher", 'context', "Method for getting next matching patch ['context']")
flags.DEFINE_float("paint_overlap_thresh", 10.0, "The threshold L2 norm error for overlapped patch areas [10.0]")
flags.DEFINE_boolean("paint_groundtruth", False, "Whether to use a groundtruth [False]")
flags.DEFINE_string("paint_groundtruth_type", 'core', "Type of groundtruth [core]")
flags.DEFINE_boolean("paint_groundtruth_new", False, "Whether to generate new groundtruth, value is passed to whatever groundtruth specified [False]")
flags.DEFINE_string("paint_groundtruth_load", None, "String specifying where to load groundtruth canvas and meta from ['None']")
flags.DEFINE_string("paint_groundtruth_save", None, "String specifying the name to save groundtruth [None]")
flags.DEFINE_integer("paint_n_cores", 0, "The number of cores to generate in the painting process, [0]")
flags.DEFINE_float("paint_core_thresh", 2.0, "The threshold L2 norm error for overlapped core areas [2.0]")
flags.DEFINE_string("paint_savefile_root", None, "String specifying the filename for saved numpy arrays and images (no file extensions) [None]")

# post sampling related flags
flags.DEFINE_boolean("post", False, "True for post sampling [False]")



# create flag object
FLAGS = flags.FLAGS

# merge flags and fixed configs into config, which gets passed to the StratGAN object
config = Config()

# training data sources
config.image_dir = os.path.join(os.pardir, 'data', FLAGS.image_dir)
config.image_ext = '*.png'
config.img_verbose = True

# model configurations
config.batch_size = FLAGS.batch_size
config.z_dim = 100                          # number inputs to gener
config.c_dim = 1
config.gf_dim = FLAGS.gf_dim                # number of gener conv filters
config.df_dim = FLAGS.df_dim                # number of discim conv filters
config.gfc_dim = 1024                       # number of gener fully connecter layer units
config.dfc_dim = 1024                       # number of discim fully connected layer units
config.alpha = 0.1                          # leaky relu alpha
config.batch_norm = True
config.minibatch_discrim = True

# training hyperparameters
config.epoch = FLAGS.epoch
config.learning_rate = FLAGS.learning_rate  # optim learn rate
config.beta1 = FLAGS.beta1                  # momentum
config.repeat_data = True
config.shuffle_data = True
config.buffer_size = 4
config.drop_remainder = True                # currently fails if false!
config.gener_iter = FLAGS.gener_iter        # times to update generator per discriminator update
config.noisy_inputs = False                 # add some small noise to the input images
config.flip_inputs = False                  # whether to flip the black white pixels

# i/o structures
config.log_dir = 'log'
config.out_dir = 'out'
config.samp_dir = 'samp'
config.chkp_dir = 'chkp'
config.paint_dir = 'paint'
config.post_dir = 'post'
config.run_dir = FLAGS.run_dir
if not config.run_dir: # if the run dir was not given, make something up
    config.run_dir = rand_id()

# painting configurations
pconfig = Config()
pconfig.label = FLAGS.paint_label
pconfig.width = FLAGS.paint_width
pconfig.height = FLAGS.paint_height
pconfig.overlap = FLAGS.paint_overlap
pconfig.patcher = FLAGS.paint_patcher
pconfig.overlap_thresh = FLAGS.paint_overlap_thresh
pconfig.groundtruth = FLAGS.paint_groundtruth
pconfig.groundtruth_type = FLAGS.paint_groundtruth_type
# pconfig.core_source = FLAGS.paint_core_source
pconfig.groundtruth_new = FLAGS.paint_groundtruth_new
pconfig.groundtruth_load = FLAGS.paint_groundtruth_load
pconfig.groundtruth_save = FLAGS.paint_groundtruth_save
pconfig.n_cores = FLAGS.paint_n_cores
pconfig.core_thresh = FLAGS.paint_core_thresh
pconfig.savefile_root = FLAGS.paint_savefile_root

# create folder structure
# -----------
folder_list = [config.out_dir, config.log_dir, 
               config.samp_dir, config.paint_dir, config.post_dir]
mkdirs(folder_list)
mkdirs([os.path.join(config.out_dir, config.run_dir),
        os.path.join(config.log_dir, config.run_dir), 
        os.path.join(config.samp_dir, config.run_dir),
        os.path.join(config.paint_dir, config.run_dir),
        os.path.join(config.post_dir, config.run_dir)]) # this should be wrapped in with mkdirs function...


# model execution function
# -----------
def main(_):

    with tf.Session() as sess:

        # build up the model, and initialize the session
        stratgan = StratGAN(sess, config)

        # if train, run the training, which saves the model
        if FLAGS.train:
            stratgan.train()

        # otherwise load
        else:
            chkp_dir = os.path.join(config.chkp_dir, config.run_dir)
            stratgan.load(chkp_dir)

        
        # now paint or do other post sampling
        if FLAGS.paint:
            stratgan.paint(pconfig)

        # elif FLAGS.context_paint:
        #     chkp_dir = os.path.join(config.chkp_dir, config.run_dir)
        #     stratgan.load(paint_chkp_dir)
        #     stratgan.context_paint()

        elif FLAGS.post:
            post_chkp_dir = os.path.join(config.chkp_dir, config.run_dir)
            stratgan.load(post_chkp_dir)
            stratgan.post_sampler(linear_interp=0, label_interp=False, random_realizations=True)

        else:
            print('Neither "train", "paint", or "post" selected. Doing nothing.')

if __name__ == '__main__':
    tf.app.run()
