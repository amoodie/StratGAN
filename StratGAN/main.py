import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils import Config
from model import StratGAN



# Setup configuration
# -----------
config = Config()
config.image_dir = os.path.join(os.pardir, 'data', 'multi_line')
config.image_ext = '*.png'
config.img_verbose = True
config.c_dim = 1

config.batch_size = 10
config.repeat_data = True
config.shuffle_data = True
config.buffer_size = 10



if not os.path.exists('out/'):
    os.makedirs('out/')



with tf.Session() as sess:
    
    # Initiate session and initialize all vaiables
    stratgan = StratGAN(sess, config)

    # sess.run([stratgan.data.image_batch, stratgan.data.label_batch])
    
    # stratgan.train()





    # i = 0
    # j = 0

    # for it in range(10):
        

    #     X_mb, _ = mnist.train.next_batch(mb_size)

        

        

        # if it % 100 == 0:
        #     print(X_mb[0].shape)
        #     fig = plot(X_mb[0])
        #     plt.savefig('X_mb/{}.png'.format(str(j).zfill(3)), bbox_inches='tight')
        #     j += 1
        #     plt.close(fig)