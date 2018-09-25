import os

class Config: 
    """
    dummy config class for storing info during generation of GUI
    """
    pass


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def mkdirs(config):
    folder_list = [config.out_dir, config.log_dir, config.samp_dir]
    for f in iter(folder_list):
        if not os.path.exists(f):
            os.makedirs(f)
