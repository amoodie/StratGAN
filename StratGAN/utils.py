import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string
import random
import json

class Config: 
    """
    dummy config class for storing info during generation of GAN/painter
    """
    pass


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot_images(images, n_categories, image_dim=None, labels=None):
    if not image_dim:
        image_dim = np.sqrt(images.shape[1])

    gd = (n_categories, images.shape[0] // n_categories) # grid image dimensions

    fig = plt.figure(figsize=(gd[1], gd[0]))
    gs = gridspec.GridSpec(gd[0], gd[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i, (image, label) in enumerate(zip(images, labels)):
        ax = plt.subplot(gs[i])
        ax.text(0.8, 0.8, str(label), 
                backgroundcolor='white', transform=ax.transAxes)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image.reshape(image_dim, image_dim), cmap='Greys_r')

    return fig


def mkdirs(folder_list):
    """makes all folders in folderlist if they don't exist"""
    for f in iter(folder_list):
        if not os.path.exists(f):
            os.makedirs(f)


def training_sample_set(z_dim, n_labels):
    n_samples = 10 # how many samples to make of each labels

    # make a set of zs to use over and over
    zs = np.random.uniform(-1, 1, [n_labels*n_samples, z_dim]).astype(np.float32)
    
    # make a set of labels to use
    idx = np.zeros((n_labels*n_samples))
    for i in np.arange(n_labels):
        cat_idx = np.tile(i, (1, n_samples))
        idx[i*n_samples:i*n_samples+n_samples] = cat_idx

    _labels = np.zeros((n_samples*n_labels, n_labels))
    _labels[np.arange(n_samples*n_labels), idx.astype(np.int)] = 1

    labels = _labels

    return zs, labels



def rand_id(size=8, chars=string.ascii_uppercase + string.digits):
    # ripped from:
    #    https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python/2257449#2257449
    return ''.join(random.choice(chars) for _ in range(size))


def write_config(model):
    atts = vars(model.config)

    with open(os.path.join(model.train_log_dir, 'config.json'), 'w') as fp:
        json.dump(atts, fp, sort_keys=True, indent=4)

def label_maker(_label, n_categories):
    """make a label (not necessarily a one hot) for evals"""

    label = np.zeros((1, n_categories))

    if isinstance(_label, (list)):
        pass 
        # thsi is where I would unpack into something meaningful
    else:
        _label = _label
        
def post_sampler():
    for i in np.arange(0, 10):
        patch = self.sess.run(self.G, feed_dict={self.z: np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(np.float32), 
                                                          self.y: np.array([[1, 0, 0, 0, 0, 0]]),
                                                          self.is_training: False})
        fig, ax = plt.subplots()
        ax.imshow(patch.squeeze(), cmap='gray')
        # plt.axis('off')
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            right=False,
            left=False,
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.savefig('post/%04d.eps' % i, bbox_inches='tight', format='eps', dpi=200)
        plt.savefig('post/%04d.png' % i, bbox_inches='tight', dpi=200)

        plt.close()