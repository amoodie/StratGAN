import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# import cv2


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
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


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
for i in np.arange(len(filelist)):
    ith = np.load(filelist[i])
    mean_array = ( (mean_array*(i+1)) + ith ) / (i + 1)



fig, ax = plt.subplots()
samp = ax.imshow(mean_array, cmap="gray")
if groundtruth:
    plt.imshow(groundtruth_canvas_overlay)
ax.axis('off')
plt.savefig(os.path.join("tmp"), bbox_inches='tight', dpi=300)
plt.close()