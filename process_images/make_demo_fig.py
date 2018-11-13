import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from PIL import Image
import numpy as np


filelist = [file for file in os.listdir('cut_images_demo') if file.endswith('.png')]

gd = (6, 3) # grid image dimensions

fig = plt.figure(figsize=(gd[1], gd[0]))
gs = gridspec.GridSpec(gd[0], gd[1])
gs.update(wspace=0.05, hspace=0.05)

labels = np.repeat([0, 1, 2, 3, 4, 5], 3)

for i, (file, label) in enumerate(zip(filelist, labels)):
    image = Image.open(os.path.join('cut_images_demo', file))
    ax = plt.subplot(gs[i])
    ax.text(0.8, 0.8, str(label), 
            backgroundcolor='white', transform=ax.transAxes)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(image, cmap='gray')

plt.savefig('input_demo.png', bbox_inches='tight')
