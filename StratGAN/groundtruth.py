import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from random import randint
import tensorflow as tf
import os
import abc




class GroundTruth(object):
    """
    Base class for adding ground truth to the core object
    """
    def __init__(self, pconfig, painter_canvas,
                 overlay_alpha=0.8):

        __metaclass__ = abc.ABCMeta

        self.pconfig = pconfig

        self.canvas_height = painter_canvas.shape[0]
        self.canvas_width = painter_canvas.shape[1]
        
        self.canvas = np.empty((self.canvas_height, self.canvas_width))
        self.canvas.fill(np.nan)
        self.canvas_overlay = np.zeros((self.canvas_height, self.canvas_width, 4))

        self.overlay_alpha = overlay_alpha

        self.type = None

        self.out_data_dir = self.pconfig.out_data_dir

        self.pconfig_to_groundtruth_source()

    def pconfig_to_groundtruth_source(self):
        if not self.pconfig.groundtruth_new and not self.pconfig.groundtruth_load:
            raise RuntimeError('must specify either new or source for groundtruth')
        if self.pconfig.groundtruth_new and self.pconfig.groundtruth_load:
            raise RuntimeError('must not specify both new and source for groundtruth')
        if self.pconfig.groundtruth_new:
            self.groundtruth_new = True
            self.groundtruth_source = self.pconfig.groundtruth_type
        elif self.pconfig.groundtruth_load:
            self.groundtruth_new = False
            self.groundtruth_source = self.pconfig.groundtruth_load
        self.groundtruth_save = self.pconfig.groundtruth_save

    def make_overlay(self):
        gt_idx = np.isfinite(self.canvas)
        channel_idx = self.canvas == 0.0
        self.canvas_overlay[np.logical_and(gt_idx, channel_idx), 0] = 61/255 # R channel, channel
        self.canvas_overlay[np.logical_and(gt_idx, np.invert(channel_idx)), 0] = 177/255 # R channel, mud
        self.canvas_overlay[np.logical_and(gt_idx, channel_idx), 1] = 116/255 # G channel
        self.canvas_overlay[np.logical_and(gt_idx, np.invert(channel_idx)), 1] = 196/255 # G channel
        self.canvas_overlay[np.logical_and(gt_idx, channel_idx), 2] = 178/255 # B channel
        self.canvas_overlay[np.logical_and(gt_idx, np.invert(channel_idx)), 2] = 231/255 # B channel
        self.canvas_overlay[gt_idx, 3] = 1 * self.overlay_alpha


class GroundTruthCores(GroundTruth):
    """
    GroundTruth object for adding cores to the painter
    """
    def __init__(self, pconfig, painter_canvas, 
                 core_width=10,
                 n_cores=None):

        GroundTruth.__init__(self, pconfig=pconfig, painter_canvas=painter_canvas)
        self.type = 'core'

        # generate any cores if needed, and quilt them into canvas
        # self.core_source = self.pconfig.core_source
        self.core_width = core_width
        
        # generate the cores by the appropriate flag
        if self.groundtruth_new:
            # if self.groundtruth_source == 'block':
            if True:
                block_height = 24
                if not n_cores:
                    self.n_cores = 1
                else:
                    self.n_cores = n_cores
                self.initialize_block_cores(n_cores=self.n_cores,
                                            n_blocks=2, block_height=block_height)
                self.meta = {'n_cores': self.n_cores}
            else:
                raise ValueError('bad core builder string given')
        else:
            print('loading core file from: ', self.groundtruth_source)
            canvas = np.load(os.path.join(self.out_data_dir, self.groundtruth_source)+'_groundtruth_canvas.npy')
            meta = np.load(os.path.join(self.out_data_dir, self.groundtruth_source)+'_groundtruth_meta.npy', allow_pickle=True)
            self.canvas = canvas
            self.meta = meta.flat[0]
            self.n_cores = self.meta['n_cores']

        # save it out (sometimes just overwrites what just got loaded)
        if self.groundtruth_save:
            np.save(os.path.join(self.out_data_dir, self.groundtruth_save+'_groundtruth_canvas.npy'), self.canvas)
            np.save(os.path.join(self.out_data_dir, self.groundtruth_save+'_groundtruth_meta.npy'), self.meta)

        self.make_overlay()

    def initialize_block_cores(self, n_cores=2, n_blocks=3, block_height=10):
        # make cores with n_blocks channel body segments. They are randomly
        # placed into the column, and may be overlapping.
        #
        # preallocate cores array, pages are cores
        core_loc = np.zeros((n_cores)).astype(np.int)
        core_val = np.zeros((self.canvas_height, self.core_width, self.n_cores))
        # make the core_val for each in n_cores
        for i in np.arange(n_cores):
            # preallocate a core matrix
            core = np.ones([self.canvas_height, self.core_width])
            # generate a random x-coordinate for top-left core corner
            ul_coord = np.random.randint(low=0, high=self.canvas_width-self.core_width, size=1)
            for j in np.arange(n_blocks):
                # generate a random y-coordinate for the top of the block
                y_coord = np.random.randint(low=0, high=self.canvas_height-block_height, size=1)[0]
                core[y_coord:y_coord+block_height, :] = 0
            # store the core into a multi-core matrix
            core_val[:, :, i] = core
            core_loc[i]       = ul_coord

        for i in np.arange(n_cores):
            self.canvas[:, core_loc[i]:core_loc[i]+self.core_width] = core_val[:,:,i]

