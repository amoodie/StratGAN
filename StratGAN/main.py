import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import utils
import ops




mb_size = 128 # minibatch size
Z_dim = 100 # 100 x 100 pixel dims?

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    stratgan = StratGAN(sess, config)


    i = 0
    j = 0

    for it in range(10):
        

        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

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

        # if it % 100 == 0:
        #     print(X_mb[0].shape)
        #     fig = plot(X_mb[0])
        #     plt.savefig('X_mb/{}.png'.format(str(j).zfill(3)), bbox_inches='tight')
        #     j += 1
        #     plt.close(fig)