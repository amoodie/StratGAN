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


def plot_images(images, dim=None, labels=None):
    if not dim:
        dim = np.sqrt(images.shape[1])

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, (image, label) in enumerate(zip(images, labels)):
        ax = plt.subplot(gs[i])
        ax.text(0.8, 0.8, str(label), transform=ax.transAxes)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image.reshape(dim, dim), cmap='Greys_r')

    return fig


def mkdirs(config):
    folder_list = [config.out_dir, config.log_dir, config.samp_dir]
    for f in iter(folder_list):
        if not os.path.exists(f):
            os.makedirs(f)
