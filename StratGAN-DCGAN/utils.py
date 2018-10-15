import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Config: 
    """
    dummy config class for storing info during generation of GUI
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


def mkdirs(config):
    folder_list = [config.out_dir, config.log_dir, config.samp_dir]
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
