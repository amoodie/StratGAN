from scipy import misc, ndimage
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np
import os


def rgb2gray(rgb):
    conv = [0.2125, 0.7154, 0.0721] # ratios to convolve with
    return np.dot(rgb[...,:3], conv)

# path of directory and list of raw images in directory
dirpath = './raw_images'
rawimgs = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]

nhslice = 10 # number of slices across deposit
nvslice = 3 # number of slices in vertical
ntslice = nhslice * nvslice # total number of slicer per image
cropdims = np.array([500, 800]) # how much to cut off top and bottom

# for i in enumerate(rawimgs):
for i in enumerate(rawimgs):
    print(i[1])

    # load image
    raw = misc.imread(os.path.join(dirpath, i[1]))

    # extract shape and use to crop image down
    lx, ly, lz = raw.shape
    crop = raw[cropdims[0] : lx-cropdims[1], :, :]

    # convert to grayscale
    gray = rgb2gray(crop)

    # binarize
    thresh = [180]
    bw = (gray > thresh)

    # dilate, erode, etc
    ero_p = ndimage.binary_opening(1.*np.invert(bw), structure=np.ones((3,3))).astype(np.int)
    ero = np.invert(ero_p)

    ero = gray
    # ero = exposure.equalize_hist(ero)
    img_adapteq = exposure.equalize_adapthist(ero, clip_limit=0.03)
    # ero = numpy.percentile(a, q, axis=None)

    # split the image into columns to loop through
    hsplt = np.array_split(ero, nhslice, 1)
    hsplt = [x for x in hsplt if x.size > 0]

    for j in enumerate(hsplt):
        
        # split the hsplt array into vertical chunks
        vsplt = np.array_split(hsplt[j[0]], nvslice, 0)
        vsplt = [x for x in vsplt if x.size > 0]

        for k in enumerate(vsplt):
            lab = (i[0]*ntslice + j[0]*nhslice + k[0])
            misc.imsave('./cut_images/%06d.png' % lab, vsplt[k[0]])

        # plt.imshow(hsplt[0], cmap='gray')
        # plt.show(block=True)

    # plt.imshow(crop)
    # plt.show(block=False)



