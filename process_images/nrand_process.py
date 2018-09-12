from scipy import misc, ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import exposure
import numpy as np
import os

# np.random.seed(seed=21548)


def rgb2gray(rgb):
    conv = [0.2125, 0.7154, 0.0721] # ratios to convolve with
    return np.dot(rgb[...,:3], conv)


def cut(image, coord, dim):
    if image.ndim > 2:
        cut = image[coord[0]:coord[0]+dim, coord[1]:coord[1]+dim, :]
    else:
        cut = image[coord[0]:coord[0]+dim, coord[1]:coord[1]+dim]
    return cut


def group_plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gd = np.ceil(np.sqrt(len(samples))).astype(np.int)
    gs = gridspec.GridSpec(gd, gd)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')

    return fig


# path of directory and list of raw images in directory
dir_path = './cropped_slices'
cropped_slices = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

n_cuts = 1000 # number of cut images to pull out of the image
cut_dim = 128 # 28 # pixels size of WxH for cut images

for i, cropped_slice in enumerate(cropped_slices):
    
    line_name = cropped_slice.rstrip('.jpg')
    print('operating on: ', line_name)

    # load image
    raw = misc.imread(os.path.join(dir_path, cropped_slice))

    # extract shape and use to crop image down
    lx, ly, lz = raw.shape # switch x and y
    # crop = raw[cropdims[0] : lx-cropdims[1], :, :]
    rx, ry = np.array([lx, ly]) - (cut_dim)
    steps_idx = np.array([np.random.randint(0, rx), np.random.randint(0, ry)])
    
    # convert to grayscale
    gray = rgb2gray(raw)

    # rescale
    # p2, p98 = np.percentile(gray, (2, 98))
    # img_rescale = exposure.rescale_intensity(gray, in_range=(p2, p98))

    # binarize
    thresh = 120 # 255 * 0.9
    bw = (gray > thresh)

    # # dilate, erode, etc
    dil_p = ndimage.binary_closing(1.*np.invert(bw), structure=np.ones((3,3))).astype(np.int)
    dil = np.invert(dil_p)

    ero_p = ndimage.binary_opening(1.*np.invert(dil), structure=np.ones((3,3))).astype(np.int)
    ero = np.invert(ero_p)

    clean = ero

    steps = [cut(raw, steps_idx, cut_dim), cut(gray, steps_idx, cut_dim), \
             cut(bw, steps_idx, cut_dim), cut(dil, steps_idx, cut_dim), cut(ero, steps_idx, cut_dim)]
    steps_fig = group_plot(steps)
    plt.savefig('out/steps_fig.png', bbox_inches='tight')
    plt.close(steps_fig)

    for j in np.arange(n_cuts):
        
        rand_idx = np.array([np.random.randint(0, rx), np.random.randint(0, ry)])
        rand_cut = cut(clean, rand_idx, cut_dim)

        lab = '%04d.png' % j
        misc.imsave(os.path.join('cut_images', lab), rand_cut)

        if j % 5 == 0:
            print('cutting image {0} of {1}'.format(j, n_cuts))
