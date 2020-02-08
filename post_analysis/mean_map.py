import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2 as cv
import shutil


def make_overlay(canvas):
    gt_idx = np.isfinite(canvas)
    channel_idx = canvas == 0.0
    canvas_overlay = np.zeros((canvas.shape[0], canvas.shape[1], 4))
    canvas_overlay[np.logical_and(gt_idx, channel_idx), 0] = 61/255 # R channel, channel
    canvas_overlay[np.logical_and(gt_idx, np.invert(channel_idx)), 0] = 177/255 # R channel, mud
    canvas_overlay[np.logical_and(gt_idx, channel_idx), 1] = 116/255 # G channel
    canvas_overlay[np.logical_and(gt_idx, np.invert(channel_idx)), 1] = 196/255 # G channel
    canvas_overlay[np.logical_and(gt_idx, channel_idx), 2] = 178/255 # B channel
    canvas_overlay[np.logical_and(gt_idx, np.invert(channel_idx)), 2] = 231/255 # B channel
    canvas_overlay[gt_idx, 3] = 1 * 0.8
    return canvas_overlay

def process_to_numpix(canvas):
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    # cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    canvas = canvas.astype(np.uint8)
    im2, contours, hierarchy = cv.findContours(canvas, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return im2, contours, hierarchy

def compute_area_index(canvas, contours, index):
    cnt = contours[0]
    M = cv.moments(cnt)
    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])
    # 2. Contour Area
    # Contour area is given by the function cv.contourArea() or from moments, M['m00'].
    area = cv.contourArea(cnt)
    return area

def canvas_plot(canvas, groundtruth_canvas_overlay, 
                filename, cmap='gray', verticies=False):
    fig, ax = plt.subplots()
    samp = ax.imshow(canvas, cmap=cmap)
    plt.imshow(groundtruth_canvas_overlay)
    ax.axis('off')
    plt.savefig(os.path.join(filename), bbox_inches='tight', dpi=300)
    plt.close()


# file list
filelist = glob.glob(os.path.join(os.path.pardir, "StratGAN", "out", "line7", "*_final.npy"))

# load ground truth array
groundtruth=True
groundtruth_canvas_overlay = np.load(os.path.join(os.path.pardir, "StratGAN", "out", "line7", "4trial_groundtruth_canvas.npy"))
groundtruth_canvas_overlay = make_overlay(groundtruth_canvas_overlay)

# open first to size objects
temp = np.load(filelist[0])
mean_array = np.zeros(temp.shape)
temp = None # clear ref



# loop to average
cp_cnt = 0
med_size = np.zeros((len(filelist)))
for i in np.arange(len(filelist)):
    
    # grab ith
    ith = np.load(filelist[i])

    # compute flood fill
    # find area flooded by known channel idx
    mask_in = np.zeros((ith.shape[0]+2,ith.shape[1]+2),np.uint8)
    corner_idxs = np.array([[210, 123], [217, 123], [210, 144], [217, 144]])
    
    mask_size = np.zeros((ith.shape[0]+2, ith.shape[1]+2, corner_idxs.shape[0]))
    # print(corner_idxs.shape)
    for j in np.arange(corner_idxs.shape[0]):
        # print(corner_idxs[1])
        num, im, mask_size[:,:,j], rect = cv.floodFill(ith.astype(np.uint8), mask_in, 
                                            (corner_idxs[j][0], corner_idxs[j][1]), 255)
    
    sum_list = np.sum(np.sum(mask_size==1,0),0)

    # find median and use this for now
    if np.all(np.median(sum_list) == sum_list) and np.median(sum_list) < 8000:
        med_size[i] = np.median(sum_list)
    else:
        med_size[i] = np.nan

    # make the mean map and image for gif
    if not np.isnan(med_size[i]):
        mean_array = ( (mean_array*(i+1)) + ith ) / (i + 1)

        # make a figure and save it for giffing
        if False:
            canvas_plot(ith, groundtruth_canvas_overlay,
                        filename=os.path.join("elite", str(cp_cnt).zfill(4)+'.png'))
            # shutil.copy(filelist[i], "elite_"+str(cp_cnt).zfill(3)+'.png')
            cp_cnt += 1

    
fig, ax = plt.subplots()
samp = ax.imshow(mean_array, cmap="gray")
if groundtruth:
    plt.imshow(groundtruth_canvas_overlay)
ax.axis('off')
plt.savefig(os.path.join("mean_array_map.png"), bbox_inches='tight', dpi=300)
plt.close()



med_size = med_size[~np.isnan(med_size)]
from scipy import stats
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
density = stats.kde.gaussian_kde(med_size)
x = np.arange(4000, 8000, 10)
vect = density(x)
ps = np.array([0.1, 0.5, 0.9])
Ps = np.interp(ps, np.cumsum(vect/np.sum(vect)), x)
Ys = np.interp(Ps, x, vect)
PsFull = np.hstack((4000, Ps, 8000)) # add on the extremes for looping
colset = [(247/255,245/255,113/255),(244/255,153/255,113/255),
          (244/255,153/255,113/255),(247/255,245/255,113/255)]

fig, ax = plt.subplots()
plt.hist(med_size, range=(4000, 8000), density=True, color='lightgray', alpha=1) #, edgecolor='black', linewidth=1.2)
ax.set_xlabel("reservoir area (px)")
ax.set_ylabel("density")
plt.savefig(os.path.join("example_size_dist.png"), bbox_inches='tight', dpi=300)

for i in np.arange(Ps.size+1):
    plt.fill_between(x[np.logical_and(x>=PsFull[i], x<=PsFull[i+1])], 0, 
                     vect[np.logical_and(x>=PsFull[i], x<=PsFull[i+1])],
                     color=colset[i], zorder=2, alpha=0.8)

for i in np.arange(Ps.size):
    plt.vlines(Ps[i], 0, Ys[i])
    plt.text(Ps[i]+50, 0.0002, "P"+str((ps[i]*100).astype(np.uint16)))

plt.plot(x, density(x), color='k')


plt.savefig(os.path.join("example_size_dist_density.png"), bbox_inches='tight', dpi=300)
plt.close()