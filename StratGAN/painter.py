import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from random import randint
import tensorflow as tf
import os


"""
A fair number of the algorithm's in this module are taken from:
https://github.com/afrozalm/Patch-Based-Texture-Synthesis
which did not carry a license at the time of use.
"""

class Painter(object):
    """
    Painter is the main object which is manipulated from the outside to
    create a realization. The Painter has an object called canvas, which
    is made up of a grid of patches which are filled through the looping. 
    """
    # all arguments are required because the defaults are handled with initial parser
    def __init__(self, stratgan,
                 paint_label, paint_width, paint_height, 
                 paint_overlap, paint_overlap_thresh, paint_core_thresh):

        print(" [*] Building painter...")

        # unpack and process inputs to convenient attributes
        self.sess = stratgan.sess
        self.stratgan = stratgan
        self.config = stratgan.config

        self.paint_samp_dir = self.stratgan.paint_samp_dir
        self.out_data_dir = self.stratgan.out_data_dir

        if not paint_label == 0 and not paint_label:
            print('Label not given for painting, assuming zero for label')
            self.paint_label = np.zeros((1, stratgan.data.n_categories))
            self.paint_label[0, 0] = 1
            self.paint_int = 0
        else:
            self.paint_label = np.zeros((1, stratgan.data.n_categories))
            self.paint_label[0, paint_label] = 1
            self.paint_int = paint_label
        self.stratgan.paint_label_tensor = self.paint_label

        self.paint_width = paint_width
        if not paint_height:
            self.paint_height = int(paint_width / 4)
        else:
            self.paint_height = paint_height

        self.overlap = paint_overlap
        self.overlap_threshold = paint_overlap_thresh

        self.core_threshold = paint_core_thresh

        self.patch_height = self.patch_width = self.config.h_dim
        self.patch_size = self.patch_height * self.patch_width

        self.canvas = Canvas(stratgan=self.stratgan,
                             canvas_width=self.paint_width, canvas_height=self.paint_height, 
                             patch_width=self.patch_width, patch_height=self.patch_height, 
                             patch_overlap=self.overlap, 
                             overlap_threshold=self.overlap_threshold,
                             core_threshold=self.core_threshold)

        # self.canvas = np.ones((self.paint_height, self.paint_width))
        
        

        # generate a random sample for the first patch and quilt into image
        # self.patch_i = 0
        # first_patch = self.generate_patch()

        # # quilt into the first coord spot
        # self.patch_coords_i = (self.patch_xcoords[self.patch_i], self.patch_ycoords[self.patch_i])
        # self.quilt_patch(self.patch_coords_i, first_patch, mcb=None)
        # self.patch_i += 1


    def add_cores(self, paint_core_source, paint_ncores):
        # generate any cores if needed, and quilt them into canvas
        self.paint_ncores = int(paint_ncores)
        self.core_source = paint_core_source
        
        self.core_width = np.int(10) # pixel width of cores

        # generate the cores by the appropriate flag
        self.core_canvas = np.zeros((self.paint_height, self.paint_width, 4))
        if self.core_source == 'new':
            block_height = 12 # pixels np.floor(self.paint_height / (12)).astype(np.int) # block size
            nblocks = 4
            self.core_val, self.core_loc = self.initialize_block_cores(nblocks=nblocks, block_height=block_height)
        elif self.core_source == 'last':
            # load the last core arrays
            self.core_val = np.load(os.path.join(self.out_data_dir, 'last_core_val.npy'))
            self.core_loc = np.load(os.path.join(self.out_data_dir, 'last_core_loc.npy'))
        else:
            if not isinstance(self.core_source, str):
                ValueError('bad core_source given, must be string')
            print('loading core file from: ', self.core_source)
            RuntimeError('not implemented yet')

        # save the cores to a "last cores" as a first step
        np.save(os.path.join(self.out_data_dir, 'last_core_val.npy'), self.core_val)
        np.save(os.path.join(self.out_data_dir, 'last_core_loc.npy'), self.core_loc)

        # quilt the cores image into the canvas and cores layer
        core_layer_alpha = 0.8
        for i in np.arange(self.paint_ncores):
            self.canvas[:, self.core_loc[i]:self.core_loc[i]+self.core_width] = self.core_val[:,:,i]
            self.core_canvas[:, self.core_loc[i]:self.core_loc[i]+self.core_width, 3] = np.ones(self.core_val[:,:,i].shape) * core_layer_alpha

        core_cmap = plt.cm.Set1
        core_idx = self.core_canvas[:, :, 3].astype(np.bool)
        channel_idx = self.canvas == 0.0
        self.core_canvas[np.logical_and(core_idx, channel_idx), 0] = 61/255 # R channel, channel
        self.core_canvas[np.logical_and(core_idx, np.invert(channel_idx)), 0] = 177/255 # R channel, mud
        self.core_canvas[np.logical_and(core_idx, channel_idx), 1] = 116/255 # G channel
        self.core_canvas[np.logical_and(core_idx, np.invert(channel_idx)), 1] = 196/255 # G channel
        self.core_canvas[np.logical_and(core_idx, channel_idx), 2] = 178/255 # B channel
        self.core_canvas[np.logical_and(core_idx, np.invert(channel_idx)), 2] = 231/255 # B channel


    def initialize_new_cores(self, nblocks, block_height):
        # make cores with nblocks channel body segments. They are
        # randomly  placed into the column, and may be overlapping.
        # block_height needs to approximately match the height of channel bodies
        #

        # preallocate cores array, pages are cores
        core_loc = np.zeros((self.paint_ncores)).astype(np.int)
        core_val = np.zeros((self.paint_height, self.core_width, self.paint_ncores))

        # make the core_val for each in ncores
        for i in np.arange(self.paint_ncores):
            # preallocate a core matrix
            core = np.ones([self.paint_height, self.core_width])

            # generate a random x-coordinate for top-left core corner
            ul_coord = np.random.randint(low=0, high=self.paint_width-self.core_width, size=1)

            for j in np.arange(nblocks):

                # generate a random y-coordinate for the top of the block
                y_coord = np.random.randint(low=0, high=self.paint_height-block_height, size=1)[0]
                core[y_coord:y_coord+block_height, :] = 0

            # store the core into a multi-core matrix
            core_val[:, :, i] = core
            core_loc[i]       = ul_coord

        return core_val, core_loc_match

    
class Canvas(object):
    """
    Canvas is the object which houses and manages the individual patches

    It keeps track of which patches have been filled, edges, cores, etc. 
    """
    # all arguments are required because the defaults are handled with initial parser
    def __init__(self, stratgan, canvas_width, canvas_height, 
                 patch_width, patch_height, patch_overlap,
                 overlap_threshold, core_threshold):

        self.stratgan = stratgan

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_overlap = patch_overlap
        self.overlap_threshold = overlap_threshold
        self.core_threshold = core_threshold
        
        # generate list of patch coordinates
        self.patch_xcoords, self.patch_ycoords = self.calculate_patch_coords()
        self.patch_count = self.patch_xcoords.size

        self.canvas = [] # initialize as an empty list
        for p in np.arange(self.patch_count):
            init_next_patch = CanvasPatch(i=p, stratgan=self.stratgan,
                                          patch_x_coord=self.patch_xcoords[p], patch_y_coord=self.patch_ycoords[p],
                                          patch_width=self.patch_width, patch_height=self.patch_height,
                                          patch_overlap=self.patch_overlap)
            self.canvas.append(init_next_patch)

    def calculate_patch_coords(self):
        """
        calculate location for patches to begin, currently ignores mod() patches
        """
        w = np.hstack((np.array([0]), np.arange(self.patch_width-self.patch_overlap, self.canvas_width-self.patch_overlap, self.patch_width-self.patch_overlap)[:-1]))
        h = np.hstack((np.array([0]), np.arange(self.patch_height-self.patch_overlap, self.canvas_height-self.patch_overlap, self.patch_height-self.patch_overlap)[:-1]))
        xm, ym = np.meshgrid(w, h)
        x = xm.flatten()
        y = ym.flatten()
        return x, y

    def realize_as_image(self):
        pass

    def fill_canvas(self):
        """
        main routine to fill out the remainder of the quilt
        """
        self.patch_i = 0 # first patch
        while self.patch_i < self.patch_count:

            self.add_next_patch(patch_number=self.patch_i)

            sys.stdout.write("     [%-20s] %-3d%%  |  [%02d]/[%d] patches  |  threshold: %2d\n" % 
                ('='*int((self.patch_i*20/self.patch_count)), int(self.patch_i/self.patch_count*100),
                 self.patch_i, self.patch_count, self.overlap_threshold_error))

            if self.patch_i % 20 == 0:
                samp = plt.imshow(self.canvas, cmap='gray')
                plt.savefig(os.path.join(self.paint_samp_dir, '%04d.png' % self.patch_i), dpi=600, bbox_inches='tight')
                plt.close()

            self.patch_i += 1

        sys.stdout.write("     [%-20s] %-3d%%  |  [%02d]/[%d] patches  |  threshold: %2d\n" % 
            ('='*int((self.patch_i*20/self.patch_count)), int(self.patch_i/self.patch_count*100),
             self.patch_i, self.patch_count, self.overlap_threshold_error))

    def add_next_patch(self, patch_number):
        """
        generate  new patch for quiliting, must pass error threshold
        """
        self.overlap_threshold_error = self.overlap_threshold
        self.core_threshold_error = self.core_threshold

        self.patch_xcoord_i = self.patch_xcoords[patch_number]
        self.patch_ycoord_i = self.patch_ycoords[patch_number]
        self.patch_coords_i = (self.patch_xcoord_i, self.patch_ycoord_i)

        match = False
        self.patch_loop = 0
        # loop until a matching patch is found, increasing thresh each time
        while not match:

            # get a new patch
            next_patch = self.canvas[patch_number]. generate_patch()

            if self.cores:
                # check for error against cores
                core_error = self.get_core_error(next_patch)

                # check patch error against core thresh
                if core_error <= self.core_threshold_error:
                    pass # continue on to check overlap error
                else:
                    self.patch_loop += 1
                    if np.mod(self.patch_loop, 100) == 0:
                        sys.stdout.write("     [%-20s] %-3d%%  |  [%02d]/[%d] patches  |  core threshold: %2d\n" % 
                            ('='*int((patch_number*20/self.patch_count)), int(patch_number/self.patch_count*100),
                            patch_number, self.patch_count, self.core_threshold_error))
                    continue # end loop iteration and try new patch

            # calculate error on the patch overlap
            patch_error, patch_error_surf = self.get_patch_error(next_patch)

            # sum/2 if it's a two-sided patch
            if len(patch_error.shape) > 0:
                patch_error = patch_error.sum() / 2

            if patch_error <= self.overlap_threshold_error:
                match = True
            else:
                self.overlap_threshold_error *= 1.01 # increase by 1% error threshold
                self.patch_loop += 1

        # calculate the minimum cost boundary
        mcb = self.calculate_min_cost_boundary(patch_error_surf)

        # then quilt it
        self.quilt_patch(self.patch_coords_i, next_patch, mcb)

        # error calculating functions:
    # ---------------------------
    def get_patch_error(self, next_patch):

        if self.patch_xcoord_i == 0:
            # a left-side patch, only calculate horizontal
            e, e_surf = self.overlap_error_horizntl(next_patch)

        elif self.patch_ycoord_i == 0:
            # a top-side patch, only calculate vertical
            e, e_surf = self.overlap_error_vertical(next_patch)

        else:
            # a center patch, calculate both
            e = np.zeros((2, 1))
            e_surf = np.zeros((2, next_patch.shape[0], next_patch.shape[1]))
            e[0], e_surf[0, 0:self.overlap, :] = self.overlap_error_horizntl(next_patch)
            e[1], e_surf[1, :, 0:self.overlap] = self.overlap_error_vertical(next_patch)

        return e, e_surf


    def overlap_error_vertical(self, next_patch):
        
        canvas_overlap = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                     self.patch_xcoord_i:self.patch_xcoord_i+self.overlap]
        patch_overlap = next_patch[:, 0:self.overlap]

        ev = np.linalg.norm(canvas_overlap - patch_overlap)
        ev_surf = (canvas_overlap - patch_overlap)**2

        return ev, ev_surf


    def overlap_error_horizntl(self, next_patch):

        canvas_overlap = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.overlap,
                                     self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]
        patch_overlap = next_patch[0:self.overlap, :]

        eh = np.linalg.norm(canvas_overlap - patch_overlap)
        eh_surf = (canvas_overlap - patch_overlap)**2
        
        return eh, eh_surf


    def get_core_error(self, next_patch):

        core_loc_match = np.logical_and(self.core_loc >= self.patch_xcoord_i,
                                   self.core_loc < self.patch_xcoord_i+self.patch_width-self.core_width)

        # check for anyting in the core list
        if np.any( core_loc_match ):
            
            core_idx = np.argmax(core_loc_match)
            # print("core_idx", core_idx)
            # print("core_loc", self.core_loc[core_idx])

            canvas_overlap = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                         self.core_loc[core_idx]:self.core_loc[core_idx]+self.core_width]
            patch_overlap = next_patch[:, self.core_loc[core_idx]-self.patch_xcoord_i:self.core_loc[core_idx]-self.patch_xcoord_i+self.core_width]

            # print("overlap_size: ", canvas_overlap.size)
            # print("num channel: ", canvas_overlap.size - np.sum(canvas_overlap))
            ec = np.linalg.norm( (patch_overlap) - (canvas_overlap))

            self.core_threshold_error = np.sqrt(  (canvas_overlap.size - np.sum(canvas_overlap)) * 0.6 ) * (1+self.patch_loop/10000)
            if self.core_threshold_error == 0.0:
                self.core_threshold_error = 6.0

            # ec = np.abs(canvas_overlap - patch_overlap).sum() / canvas_overlap.size
            # print('calculated: ', ec, '; threshold: ', self.core_threshold_error, '; theoretical max: ', np.linalg.norm( canvas_overlap - (1-canvas_overlap) ))


            if ec <= self.core_threshold_error:
                # self.dbfig, self.dbax = plt.subplots(2, 2)
                # self.dbax[0, 0].imshow(self.canvas, cmap='gray')
                # self.dbax[0, 0].plot(self.patch_xcoord_i, self.patch_ycoord_i, 'r*')
                # for i, s in enumerate(self.core_loc):
                #     self.dbax[0, 0].annotate(i, (self.core_loc[i], 250))
                # self.dbax[0, 1].imshow( (canvas_overlap - patch_overlap)**2, cmap='gray')
                # self.dbax[1, 0].imshow(next_patch, cmap='gray')
                # self.dbax[1, 0].add_patch(patches.Rectangle((self.core_loc[core_idx]-self.patch_xcoord_i, 0), self.core_width, self.patch_height,
                #                                             facecolor='none', edgecolor='red'))
                # canvas_area = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                #                           self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]
                # self.dbax[1, 1].imshow(canvas_area, cmap='gray')
                # self.dbax[1, 1].add_patch(patches.Rectangle((self.core_loc[core_idx]-self.patch_xcoord_i, 0), self.core_width, self.patch_height,
                #                                             facecolor='none', edgecolor='red'))
                
                # plt.show()
                pass

        else:
            ec = 0.0

        return ec


    # min cost boundary functions:
    # ----------------------------
    def calculate_min_cost_boundary(self, patch_error_surf):

        if self.patch_xcoord_i == 0:
            # a left-side patch, only calculate horizontal
            mcb = self.min_cost_path_horizntl(patch_error_surf)

        elif self.patch_ycoord_i == 0:
            # a top-side patch, only calculate vertical
            mcb = self.min_cost_path_vertical(patch_error_surf)

        else:
            # a center patch, calculate both
            assert patch_error_surf.shape[1] == patch_error_surf.shape[2] # this will fail if patch not square
            mcb = np.zeros((2, patch_error_surf.shape[1]), np.int8)
            # print("mcb shape:", mcb.shape)
            mcb[0, :] = self.min_cost_path_horizntl(patch_error_surf[0, 0:self.overlap, :])
            mcb[1, :] = self.min_cost_path_vertical(patch_error_surf[1, :, 0:self.overlap])

        return mcb


    def min_cost_path_vertical(self, patch_error_surf):
        mcb = np.zeros((self.patch_height), np.int) # mincostboundary
        holder = np.zeros((self.patch_height, self.overlap), np.int) # holder matrix for temp storage
        for i in np.arange(1, self.patch_height):
            # for the height of the patch
            for j in np.arange(self.overlap):
                # for each col in overlap
                if j == 0:
                    # if first col in row
                    holder[i,j] = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j+1] else j+1
                elif j == self.overlap - 1:
                    # if last col in row
                    holder[i,j] = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j-1] else j-1
                else:
                    # if center cols
                    curr_min = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j-1] else j-1
                    holder[i,j] = curr_min if patch_error_surf[i-1,curr_min] < patch_error_surf[i-1,j+1] else j+1
                
                patch_error_surf[i,j] += patch_error_surf[i-1, holder[i,j]]

        min_idx = 0
        for j in np.arange(1, self.overlap):
            min_idx = min_idx if patch_error_surf[self.patch_height - 1, min_idx] < patch_error_surf[self.patch_height - 1, j] else j
        
        mcb[self.patch_height-1] = min_idx
        for i in np.arange(self.patch_height - 1, 0, -1):
            mcb[i - 1] = holder[i, mcb[i]]

        return mcb


    def min_cost_path_horizntl(self, patch_error_surf):
        mcb = np.zeros((self.patch_width), np.int) # mincostboundary
        holder = np.zeros((self.overlap, self.patch_width), np.int)
        for j in np.arange(1, self.patch_width):
            for i in np.arange(self.overlap):
                if i == 0:
                    holder[i,j] = i if patch_error_surf[i,j-1] < patch_error_surf[i+1,j-1] else i + 1
                
                elif i == self.overlap - 1:
                    holder[i,j] = i if patch_error_surf[i,j-1] < patch_error_surf[i-1,j-1] else i - 1
                
                else:
                    curr_min = i if patch_error_surf[i,j-1] < patch_error_surf[i-1,j-1] else i - 1
                    holder[i,j] = curr_min if patch_error_surf[curr_min,j-1] < patch_error_surf[i-1,j-1] else i + 1
                
                patch_error_surf[i,j] += patch_error_surf[holder[i,j], j-1]

        min_idx = 0
        for i in np.arange(1,self.overlap):
            min_idx = min_idx if patch_error_surf[min_idx, self.patch_width - 1] < patch_error_surf[i, self.patch_width - 1] else i
        
        mcb[self.patch_width-1] = min_idx
        for j in np.arange(self.patch_width - 1,0,-1):
            mcb[j - 1] = holder[mcb[j],j]
        
        return mcb


class CanvasPatch(object):
    """
    CanvasPatch object which comprises one element of the main Canvas object.

    Methods:

    """
    def __init__(self, i, stratgan, patch_x_coord, patch_y_coord, 
                 patch_width, patch_height, patch_overlap):
        
        self.i = i
        self.stratgan = stratgan
        self.patch_x_coord = patch_x_coord
        self.patch_y_coord = patch_y_coord
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_overlap = patch_overlap

        self.is_filled = False

        # main patch attribute which holds information as 0 or 1
        self.patch = np.zeros((self.patch_height, self.patch_width))

        # preallocate and then reassign index parts
        #   might stick this in submethod with options to handle partial patches
        self.upper_idx = self.lower_idx = self.left_idx = \
            self.right_idx = self.center_idx = np.full((self.patch_height, self.patch_width), False)
        self.upper_idx[:self.patch_overlap, :] = True
        # .... 
        # ....
    
    def generate_patch(self):
        # use the GAN to make a patch
        z = np.random.uniform(-1, 1, [1, self.stratgan.config.z_dim]).astype(np.float32)
        paint_label_tensor = self.stratgan.paint_label_tensor
        patch = self.stratgan.sess.run(self.stratgan.G, feed_dict={self.stratgan.z: z, 
                                                 self.stratgan.y: paint_label_tensor,
                                                 self.stratgan.is_training: False})
        r_patch = patch[0].reshape(self.stratgan.config.h_dim, self.stratgan.config.h_dim)
        return r_patch

    def realize_as_image(self):
        self.compose_parts()
        pass

    def compose_parts(self):
        pass

    def fill_upper():
        pass




    # Quilting Functions:
    # -------------------
    def quilt_patch(self, coords, patch, mcb=None):

        if mcb is None:
            # print("NO MCB")
            y = coords[0]
            x = coords[1]
            self.canvas[x:x+self.patch_height, y:y+self.patch_width] = np.squeeze(patch)
        else:

            if self.patch_xcoord_i == 0:
                # a left-side patch, only calculate horizontal
                # with fresh canvas
                # self.dbfig, self.dbax = plt.subplots(2, 2)
                # self.dbax[0, 0].imshow(self.canvas, cmap='gray')
                # self.dbax[0, 0].plot(self.patch_xcoord_i + np.arange(self.patch_width), self.patch_ycoord_i + mcb, 'r')
                # self.dbax[0, 1].imshow(patch, cmap='gray')
                # self.dbax[0, 1].plot(np.arange(self.patch_width), mcb, 'r')

                self.quilt_overlap_horizntl(coords, patch, mcb)

                # with overlap
                # self.dbax[1, 0].imshow(self.canvas, cmap='gray')
                # self.dbax[1, 0].plot(self.patch_xcoord_i + np.arange(self.patch_width), self.patch_ycoord_i + mcb, 'r')


                self.quilt_patch_remainder(coords, patch, switch='h')

                # with remainder
                # self.dbax[1, 1].imshow(self.canvas, cmap='gray')
                # self.dbax[1, 1].plot(self.patch_xcoord_i + np.arange(self.patch_width), self.patch_ycoord_i + mcb, 'r')
                # plt.show()

            elif self.patch_ycoord_i == 0:
                # a top-side patch, only calculate vertical

                self.quilt_overlap_vertical(coords, patch, mcb)
                
                self.quilt_patch_remainder(coords, patch, switch='v')

            else:
                # a center patch, calculate both
                self.quilt_overlap_horizntl(coords, patch, mcb[0, :])
                self.quilt_overlap_vertical(coords, patch, mcb[1, :])
                self.quilt_patch_remainder(coords, patch, switch='b')
  

    def quilt_overlap_vertical(self, coords, patch, mcb):
        y = coords[0]
        x = coords[1]
        for i in np.arange(self.patch_height):
            # for each row in the overlap
            for j in np.arange(mcb[i], self.overlap):
                # for each column beyond the mcb
                self.canvas[x+i, y+j] = patch[i, j]


    def quilt_overlap_horizntl(self, coords, patch, mcb):
        y = coords[0]
        x = coords[1]
        
        for i in np.arange(self.patch_width):
            # for each column in the overlap
            for j in np.arange(mcb[i], self.overlap):
                # for each row below mcb
                self.canvas[x+j, y+i] = patch[j, i]


    def quilt_patch_remainder(self, coords, patch, switch):
        y = coords[0]
        x = coords[1]

        if switch == 'h':
            x0 = x+self.overlap
            patch_remainder = patch[self.overlap:, :]
            self.canvas[x0:x+self.patch_width, y:y+self.patch_height] = np.squeeze(patch_remainder)
        
        elif switch == 'v':
            y0 = y+self.overlap
            patch_remainder = patch[:, self.overlap:]
            self.canvas[x:x+self.patch_width, y0:y+self.patch_height] = np.squeeze(patch_remainder)

        elif switch == 'b':
            y0 = y+self.overlap
            x0 = x+self.overlap
            patch_remainder = patch[self.overlap:, self.overlap:]
            self.canvas[x0:x+self.patch_width, y0:y+self.patch_height] = np.squeeze(patch_remainder)

