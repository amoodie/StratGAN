import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np
from random import randint
import tensorflow as tf
import os
import abc


"""
A fair number of the algorithm's in this module are taken from:
https://github.com/afrozalm/Patch-Based-Texture-Synthesis
Which did not carry a license at the time of use.
"""

class CanvasPainter(object):
    """
    defines all methods for quilting,
    superclasses just overwrite the next_patch method
    """
    def __init__(self, stratgan, paint_label=None, 
                 canvas_width=1000, canvas_height=None, 
                 patch_overlap=24, batch_dim=1):

        print(" [*] Building painter...")

        __metaclass__ = abc.ABCMeta

        self.sess = stratgan.sess
        self.stratgan = stratgan
        self.config = stratgan.config

        self.paint_samp_dir = self.stratgan.paint_samp_dir
        self.out_data_dir = self.stratgan.out_data_dir

        self.batch_dim = batch_dim

        if not paint_label == 0 and not paint_label:
            print('Label not given for painting, assuming zero for label')
            self.paint_label = np.zeros((self.batch_dim, stratgan.data.n_categories))
            self.paint_label[:, 0] = 1
            self.paint_label_int = 0
        else:
            # label = tf.one_hot(label, self.config.n_categories)
            self.paint_label = np.zeros((self.batch_dim, stratgan.data.n_categories))
            self.paint_label[:, paint_label] = 1
            self.paint_label_int = paint_label

        # dump the input canvas size etc into fields
        self.canvas_width = canvas_width
        if not canvas_height:
            self.canvas_height = int(canvas_width / 4)
        else:
            self.canvas_height = canvas_height
        self.patch_overlap = patch_overlap
        self.patch_height = self.patch_width = self.config.h_dim
        self.patch_numel = self.patch_height * self.patch_width

        # generate the list of patch coordinates
        self.patch_xcoords, self.patch_ycoords = self.calculate_patch_coords()
        self.patch_count = self.patch_xcoords.size

        # cull down the canvas size to match (orphan boundaries)
        self.canvas_width = self.patch_xcoords[-1] + self.patch_width
        self.canvas_height = self.patch_ycoords[-1] + self.patch_height

        self.canvas = np.ones((self.canvas_height, self.canvas_width))
        self.target_canvas = 0.5 * np.ones((self.canvas_height, self.canvas_width))
        self.quilted_canvas = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)



        # by default there is no ground truth objects
        # self.groundtruth_type = None


    def calculate_patch_coords(self):
        """
        calculate location for patches to begin, currently ignores mod() patches
        """
        w = np.hstack((np.array([0]), np.arange(self.patch_width-self.patch_overlap,
                                                self.canvas_width-self.patch_overlap,
                                                self.patch_width-self.patch_overlap)[:-1]))
        h = np.hstack((np.array([0]), np.arange(self.patch_height-self.patch_overlap,
                                                self.canvas_height-self.patch_overlap,
                                                self.patch_height-self.patch_overlap)[:-1]))
        xm, ym = np.meshgrid(w, h)
        x = xm.flatten()
        y = ym.flatten()
        return x, y


    def add_next_patch(self, calculate_mcb=True):
        """
        find new patch for quiliting, must pass error threshold
        """
        self.patch_xcoord_i = self.patch_xcoords[self.patch_i]
        self.patch_ycoord_i = self.patch_ycoords[self.patch_i]
        self.patch_coords_i = (self.patch_xcoord_i, self.patch_ycoord_i)

        next_patch = self.generate_next_patch()
        _, patch_error_surf = self.calculate_patch_error_surf(next_patch)

        # calculate the minimum cost boundary
        if calculate_mcb:
            mcb = self.calculate_min_cost_boundary(patch_error_surf)
        else:
            mcb = None

        # then quilt it
        self.quilt_patch(self.patch_coords_i, next_patch, mcb)
        self.patch_i += 1
        

    def fill_canvas(self):
        # generate a random sample for the first patch and quilt into image
        # first_patch = self.generate_next_patch()

        # # quilt into the first coord spot
        # self.patch_coords_i = (self.patch_xcoords[self.patch_i], self.patch_ycoords[self.patch_i])
        # self.quilt_patch(self.patch_coords_i, first_patch, mcb=None)
        print("filling")
        self.patch_i = 0
        self.add_next_patch(calculate_mcb=False)


        # main routine to fill out the remainder of the quilt
        while self.patch_i < self.patch_count:

            self.add_next_patch()


            sys.stdout.write("     [%-20s] %-3d%%  |  [%02d]/[%d] patches\n" % 
                ('='*int((self.patch_i*20/self.patch_count)), int(self.patch_i/self.patch_count*100),
                 self.patch_i, self.patch_count))

            if self.patch_i % 20 == 0:
                samp = plt.imshow(self.canvas, cmap='gray')
                plt.savefig(os.path.join(self.paint_samp_dir, '%04d.png' % self.patch_i), dpi=600, bbox_inches='tight')
                plt.close()

            

        sys.stdout.write("     [%-20s] %-3d%%  |  [%02d]/[%d] patches\n" % 
            ('='*int((self.patch_i*20/self.patch_count)), int(self.patch_i/self.patch_count*100),
             self.patch_i, self.patch_count))


    @abc.abstractmethod
    def generate_next_patch(self, **kwargs):
        """
        abstract method for generating the next patch,
        must be implemented in subclass
        """
        pass


    def add_groundtruth(self, groundtruth):
        if groundtruth.canvas.shape != self.canvas.shape:
            RuntimeError('ground truth must have common shape with canvas')

        self.groundtruth_obj = groundtruth
        self.groundtruth_canvas = np.copy(groundtruth.canvas)
        self.groundtruth_canvas_overlay = np.copy(groundtruth.canvas_overlay)        
        self.groundtruth_type = groundtruth.type
        self.groundtruth = True

    # error surface calculations:
    # ----------------------------
    def calculate_patch_error_surf(self, next_patch):
        if self.patch_xcoord_i == 0:
            # a left-side patch, only calculate horizontal
            e, e_surf = self.patch_overlap_error_horizntl(next_patch)

        elif self.patch_ycoord_i == 0:
            # a top-side patch, only calculate vertical
            e, e_surf = self.patch_overlap_error_vertical(next_patch)

        else:
            # a center patch, calculate both
            e = np.zeros((2, 1))
            e_surf = np.zeros((2, next_patch.shape[0], next_patch.shape[1]))
            e[0], e_surf[0, 0:self.patch_overlap, :] = self.patch_overlap_error_horizntl(next_patch)
            e[1], e_surf[1, :, 0:self.patch_overlap] = self.patch_overlap_error_vertical(next_patch)

        return e, e_surf


    def patch_overlap_error_vertical(self, next_patch):
        
        canvas_overlaped = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                     self.patch_xcoord_i:self.patch_xcoord_i+self.patch_overlap]
        patch_overlaped = next_patch[:, 0:self.patch_overlap]

        ev = np.linalg.norm(canvas_overlaped - patch_overlaped)
        ev_surf = (canvas_overlaped - patch_overlaped)**2

        return ev, ev_surf


    def patch_overlap_error_horizntl(self, next_patch):

        canvas_overlaped = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_overlap,
                                     self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]
        patch_overlaped = next_patch[0:self.patch_overlap, :]

        eh = np.linalg.norm(canvas_overlaped - patch_overlaped)
        eh_surf = (canvas_overlaped - patch_overlaped)**2
        
        return eh, eh_surf

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
            mcb[0, :] = self.min_cost_path_horizntl(patch_error_surf[0, 0:self.patch_overlap, :])
            mcb[1, :] = self.min_cost_path_vertical(patch_error_surf[1, :, 0:self.patch_overlap])

        return mcb


    def min_cost_path_vertical(self, patch_error_surf):
        mcb = np.zeros((self.patch_height), np.int) # mincostboundary
        holder = np.zeros((self.patch_height, self.patch_overlap), np.int) # holder matrix for temp storage
        for i in np.arange(1, self.patch_height):
            # for the height of the patch
            for j in np.arange(self.patch_overlap):
                # for each col in overlap
                if j == 0:
                    # if first col in row
                    holder[i,j] = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j+1] else j+1
                elif j == self.patch_overlap - 1:
                    # if last col in row
                    holder[i,j] = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j-1] else j-1
                else:
                    # if center cols
                    curr_min = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j-1] else j-1
                    holder[i,j] = curr_min if patch_error_surf[i-1,curr_min] < patch_error_surf[i-1,j+1] else j+1
                
                patch_error_surf[i,j] += patch_error_surf[i-1, holder[i,j]]

        min_idx = 0
        for j in np.arange(1, self.patch_overlap):
            min_idx = min_idx if patch_error_surf[self.patch_height - 1, min_idx] < patch_error_surf[self.patch_height - 1, j] else j
        
        mcb[self.patch_height-1] = min_idx
        for i in np.arange(self.patch_height - 1, 0, -1):
            mcb[i - 1] = holder[i, mcb[i]]

        return mcb


    def min_cost_path_horizntl(self, patch_error_surf):
        mcb = np.zeros((self.patch_width), np.int) # mincostboundary
        holder = np.zeros((self.patch_overlap, self.patch_width), np.int)
        for j in np.arange(1, self.patch_width):
            for i in np.arange(self.patch_overlap):
                if i == 0:
                    holder[i,j] = i if patch_error_surf[i,j-1] < patch_error_surf[i+1,j-1] else i + 1
                elif i == self.patch_overlap - 1:
                    holder[i,j] = i if patch_error_surf[i,j-1] < patch_error_surf[i-1,j-1] else i - 1
                else:
                    curr_min = i if patch_error_surf[i,j-1] < patch_error_surf[i-1,j-1] else i - 1
                    holder[i,j] = curr_min if patch_error_surf[curr_min,j-1] < patch_error_surf[i-1,j-1] else i + 1
                
                patch_error_surf[i,j] += patch_error_surf[holder[i,j], j-1]

        min_idx = 0
        for i in np.arange(1,self.patch_overlap):
            min_idx = min_idx if patch_error_surf[min_idx, self.patch_width - 1] < patch_error_surf[i, self.patch_width - 1] else i
        
        mcb[self.patch_width-1] = min_idx
        for j in np.arange(self.patch_width - 1,0,-1):
            mcb[j - 1] = holder[mcb[j],j]
        
        return mcb


    # Quilting Functions:
    # -------------------
    def quilt_patch(self, coords, patch, mcb=None):
        y = coords[0]
        x = coords[1]
        if mcb is None:
            # first patch, or set to ignore all mcbs
            self.canvas[x:x+self.patch_height, y:y+self.patch_width] = np.squeeze(patch)
            self.target_canvas[x:x+self.patch_height, y:y+self.patch_width] = np.squeeze(patch)
            # self.quilted_canvas[x:x+self.patch_height, y:y+self.patch_width] = True
        else:
            if self.patch_xcoord_i == 0:
                # a left-side patch, over calculate horizontal
                self.quilt_overlap_horizntl(coords, patch, mcb)
                self.quilt_patch_remainder(coords, patch, switch='h')
            elif self.patch_ycoord_i == 0:
                # a top-side patch, only calculate vertical
                self.quilt_overlap_vertical(coords, patch, mcb)
                self.quilt_patch_remainder(coords, patch, switch='v')
            else:
                # a center patch, calculate both
                self.quilt_overlap_horizntl(coords, patch, mcb[0, :])
                self.quilt_overlap_vertical(coords, patch, mcb[1, :])
                self.quilt_patch_remainder(coords, patch, switch='b')
        self.quilted_canvas[x:x+self.patch_width, y:y+self.patch_height] = True
  

    def quilt_overlap_vertical(self, coords, patch, mcb):
        y = coords[0]
        x = coords[1]
        for i in np.arange(self.patch_height):
            # for each row in the overlap
            for j in np.arange(mcb[i], self.patch_overlap):
                # for each column beyond the mcb
                self.canvas[x+i, y+j] = patch[i, j]
                self.target_canvas[x+i, y+j] = patch[i, j]
                # self.quilted_canvas[x+i, y+j] = patch[i, j]


    def quilt_overlap_horizntl(self, coords, patch, mcb):
        y = coords[0]
        x = coords[1]
        for i in np.arange(self.patch_width):
            # for each column in the overlap
            for j in np.arange(mcb[i], self.patch_overlap):
                # for each row below mcb
                self.canvas[x+j, y+i] = patch[j, i]
                self.target_canvas[x+j, y+i] = patch[j, i]
                # self.quilted_canvas[x+j, y+i] = True


    def quilt_patch_remainder(self, coords, patch, switch):
        y = coords[0]
        x = coords[1]
        if switch == 'h':
            x0 = x+self.patch_overlap
            patch_remainder = patch[self.patch_overlap:, :]
            self.canvas[x0:x+self.patch_width, y:y+self.patch_height] = np.squeeze(patch_remainder)
            self.target_canvas[x0:x+self.patch_width, y:y+self.patch_height] = np.squeeze(patch_remainder)
        elif switch == 'v':
            y0 = y+self.patch_overlap
            patch_remainder = patch[:, self.patch_overlap:]
            self.canvas[x:x+self.patch_width, y0:y+self.patch_height] = np.squeeze(patch_remainder)
            self.target_canvas[x:x+self.patch_width, y0:y+self.patch_height] = np.squeeze(patch_remainder)
            # self.quilted_canvas[x:x+self.patch_width, y0:y+self.patch_height] = True
        elif switch == 'b':
            y0 = y+self.patch_overlap
            x0 = x+self.patch_overlap
            patch_remainder = patch[self.patch_overlap:, self.patch_overlap:]
            self.canvas[x0:x+self.patch_width, y0:y+self.patch_height] = np.squeeze(patch_remainder)
            self.target_canvas[x0:x+self.patch_width, y0:y+self.patch_height] = np.squeeze(patch_remainder)
            # self.quilted_canvas[x0:x+self.patch_width, y0:y+self.patch_height] = True


    def canvas_plot(self, filename, cmap='gray', verticies=False):
        fig, ax = plt.subplots()
        samp = ax.imshow(self.canvas, cmap=cmap)
        if self.groundtruth:
            plt.imshow(self.groundtruth_canvas_overlay)
        ax.axis('off')
        if verticies:
            plt.plot(self.patch_xcoords, self.patch_ycoords, marker='.', ls='none', ms=2)
        plt.savefig(os.path.join(self.paint_samp_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()




class ContextPainter(CanvasPainter):
    def __init__(self, stratgan, paint_label, 
                 canvas_width, canvas_height,
                 patch_overlap, patch_overlap_threshold,
                 batch_dim=40):
        CanvasPainter.__init__(self, stratgan=stratgan,
                               paint_label=paint_label, 
                               canvas_width=canvas_width, 
                               canvas_height=canvas_height, 
                               patch_overlap=patch_overlap,
                               batch_dim=batch_dim) 

        print(" [*] Building painter...")

        graph = tf.get_default_graph()
        self.gi = graph.get_tensor_by_name('gener/g_in:0')
        self.go = graph.get_tensor_by_name('gener/g_prob:0')
        self.do = graph.get_tensor_by_name('discr_1/Sigmoid:0')
        self.gl = tf.log(1 - self.do)
        
        self.build_input_placeholders()
        self.build_context_loss()
        self.lam = 2. # weighting for realism
        self.gam = 0.2 # adjustment for non-ground truth context

        self.perceptual_loss = self.gl
        self.inpaint_loss = self.context_loss + self.lam*self.perceptual_loss
        self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)

        self.img_cntr = 0

      
    def build_context_loss(self):
        """Builds the context loss objective"""
        self.go = tf.reshape(self.go, [self.batch_dim, -1])
        self.context_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.masks, self.go) -
                       tf.multiply(self.masks, self.targets))), 1)


    def build_input_placeholders(self):
      # with self.graph.as_default():
        self.masks = tf.placeholder(tf.float32,
                                    (self.batch_dim, self.patch_height*self.patch_width),
                                    name='masks')
        self.targets = tf.placeholder(tf.float32,
                                     (self.batch_dim, self.patch_height*self.patch_width),
                                     name='targets')

    def generate_next_patch(self):

        self.extract_context_mask()

        v = 0.1
        momentum = 1
        lr = 0.001

        self.z_in = np.random.normal(-1, 1, [self.batch_dim, self.config.z_dim]).astype(np.float32)
        
        # self.writer = tf.summary.FileWriter(self.stratgan.train_log_dir,
                                            # graph=self.sess.graph)
        # self.writer.flush()
        for i in np.arange(50):
            # out_vars = [self.stratgan.G, self.inpaint_loss, self.inpaint_grad]
            in_dict={self.stratgan.z: self.z_in, 
                     self.stratgan.y: self.paint_label,
                     self.stratgan.is_training: False,
                     self.masks: self.masks0,
                     self.targets: self.targets0}
          
            out_vars = [self.inpaint_loss, self.inpaint_grad, self.go]
            loss, grad, patch = self.sess.run(out_vars, feed_dict=in_dict)

            if False and np.mod(i, 10)==0:
                patch_reshaped = np.reshape(patch, (self.batch_dim, \
                                                self.patch_width, self.patch_height))
                ptch = []
                for p in np.arange(3):
                    ptch.append( patches.Rectangle((self.patch_xcoord_i, self.patch_ycoord_i),
                                              width=self.patch_width, height=self.patch_height,
                                              edgecolor='r', facecolor='None') )
                fig = plt.figure()
                # fig.subplots_adjust(hspace=0.025, wspace=0.025)
                gs = fig.add_gridspec(4, 4)
                ax1 = fig.add_subplot(gs[0,1:3])
                cnv = ax1.imshow(self.canvas, cmap='gray')
                if self.groundtruth:
                    ax1.imshow(self.groundtruth_canvas_overlay)
                ax1.add_patch(ptch[0])
                cnv.set_clim(0.0, 1.0)
                ax1.axes.xaxis.set_ticklabels([])
                ax1.axes.yaxis.set_ticklabels([])
                ax2 = fig.add_subplot(gs[1,1:3])
                tcnv = ax2.imshow(self.target_canvas, cmap='gray')
                ax2.add_patch(ptch[1])
                tcnv.set_clim(0.0, 1.0)
                ax2.axes.xaxis.set_ticklabels([])
                ax2.axes.yaxis.set_ticklabels([])
                ax3 = fig.add_subplot(gs[2,1:3])
                qcnv = ax3.imshow(self.quilted_canvas, cmap='gray')
                ax3.add_patch(ptch[2])
                qcnv.set_clim(0.0, 1.0)
                ax3.axes.xaxis.set_ticklabels([])
                ax3.axes.yaxis.set_ticklabels([])
                ax5 = fig.add_subplot(gs[1,0])
                tgt = ax5.imshow(self.target_as_image, cmap='gray')
                tgt.set_clim(0.0, 1.0)
                ax5.axes.xaxis.set_ticklabels([])
                ax5.axes.yaxis.set_ticklabels([])
                ax4 = fig.add_subplot(gs[2,0])
                msk = ax4.imshow(self.mask_as_image, cmap='gray')
                msk.set_clim(0.0, 1.0)
                ax4.axes.xaxis.set_ticklabels([])
                ax4.axes.yaxis.set_ticklabels([])
                ax6 = fig.add_subplot(gs[:2,3])
                zs = ax6.imshow(self.z_in.T)
                zs.set_clim(-1.0, 1.0)
                ax6.axes.xaxis.set_ticklabels([])
                ax6.axes.yaxis.set_ticklabels([])
                ax6.set_xlabel('batch')
                r = 3
                adj = 0
                for o, p in enumerate( np.random.randint(low=0, high=self.batch_dim, size=(4)) ):
                    # if o>=r: 
                    #     r = 4
                    #     adj = 4
                    axp = fig.add_subplot(gs[r,o])
                    ptch = axp.imshow(patch_reshaped[p,:,:], cmap='gray')
                    ptch.set_clim(0.0, 1.0)
                    axp.axes.xaxis.set_ticklabels([])
                    axp.axes.yaxis.set_ticklabels([])


                # plt.savefig(os.path.join(self.paint_samp_dir, 'context_i.png'), 
                plt.savefig(os.path.join(self.paint_samp_dir, 'iters/context_{0}.png'.format(str(self.img_cntr).zfill(4))), 
                            bbox_inches='tight', dpi=150, transparent=False)
                plt.close()
                self.img_cntr += 1

            v_prev = np.copy(v)
            v = momentum*v - lr*grad[0]
            self.z_in += (-momentum * v_prev +
                     (1 + momentum) * v)
            self.z_in = np.clip(self.z_in, -1, 1)

            verbose = False
            if verbose:
                print('Iteration {}: {}'.format(i, np.mean(loss)))
            
        # routine for determining the patch from the batch
        # print("shape:", self.context_loss.shape)
        # min_loc = np.argmin(np.mean(loss,1))
        min_loc = np.argmin(self.context_loss)

        next_patch = np.copy(patch[min_loc,:])
        next_patch = next_patch.reshape(self.config.h_dim, self.config.h_dim)
        return next_patch
    

    def extract_context_mask(self):
        """
        extract the target image and the corresponging mask
        """
        target_extract = self.target_canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                            self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]
        quilted_extract = self.quilted_canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                              self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]

        # expand and modify the target and mask
        mask_tilde = np.copy(quilted_extract)
        target_tilde = np.copy(target_extract)
        
        target_tilde -= 0.5
        target_tilde *= self.gam
        target_tilde += 0.5

        # import groundtruth information
        if self.groundtruth:
            groundtruth_extract = self.groundtruth_canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                                          self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]
            has_truth = np.isfinite(groundtruth_extract)

            target_tilde[has_truth] = groundtruth_extract[has_truth]
            mask_tilde[has_truth] = 1

        # reshape and store in feeds
        target_flat = target_tilde.reshape(1, -1)
        self.targets0 = np.tile(target_flat, (self.batch_dim, 1))
        
        mask_flat = mask_tilde.flatten()
        self.masks0 = np.zeros((self.batch_dim, self.patch_width*self.patch_height), dtype=np.float32)
        self.masks0[:, mask_flat] = 1.

        self.patches0 = np.zeros((1, self.patch_width, self.patch_height), dtype=np.float32)

        # convert the masks to images for plotting
        self.mask_as_image = np.reshape(self.masks0[0,:], 
                                (self.patch_width, self.patch_height))
        self.target_as_image = np.reshape(self.targets0[0,:], 
                                (self.patch_width, self.patch_height))
        self.patch0_as_image = np.reshape(self.patches0, 
                                (self.patch_width, self.patch_height))







class EfrosPainter(CanvasPainter):
    def __init__(self, stratgan, paint_label, 
                 canvas_width, canvas_height,
                 patch_overlap, patch_overlap_threshold,
                 ground_truth_weight=8):
        CanvasPainter.__init__(self, stratgan=stratgan,
                               paint_label=paint_label, 
                               canvas_width=canvas_width, 
                               canvas_height=canvas_height, 
                               patch_overlap=patch_overlap)

        self.patch_overlap_threshold = patch_overlap_threshold
        self.ground_truth_weight = ground_truth_weight
        self.paint_batch_size = 1

    def generate_next_patch(self):
        
        patch_overlap_threshold_this_patch = self.patch_overlap_threshold
        
        self.match = False
        self.patch_loop = 0

        # loop until a matching patch is found, increasing thresh each time
        while not self.match:
            # get a new patch
            next_patch = self.generate_random_patch()

            if self.groundtruth_cores:
                # check for error against cores
                core_error = self.get_core_error(next_patch)

                # check patch error against core thresh
                if core_error <= self.core_threshold_error: 
                    pass # continue on to check overlap error
                else:
                    self.patch_loop += 1
                    if np.mod(self.patch_loop, 100) == 0:
                        sys.stdout.write("     [%-20s] %-3d%%  |  [%02d]/[%d] patches  |  core threshold: %2d\n" % 
                            ('='*int((self.patch_i*20/self.patch_count)), int(self.patch_i/self.patch_count*100),
                            self.patch_i, self.patch_count, self.core_threshold_error))
                    continue # end loop iteration and try new patch

            # calculate error on the patch overlap
            patch_error, patch_error_surf = self.calculate_patch_error_surf(next_patch)

            # sum/2 if it's a two-sided patch
            if len(patch_error.shape) > 0:
                patch_error = patch_error.sum() / 2

            if patch_error <= patch_overlap_threshold_this_patch:
                self.match = True
            else:
                patch_overlap_threshold_this_patch *= 1.01 # increase by 1% error threshold
                self.patch_loop += 1

        return next_patch


    def generate_random_patch(self):
        # use the GAN to make a random guess patch
        z = np.random.uniform(-1, 1, [self.paint_batch_size, self.config.z_dim]).astype(np.float32)
        paint_label = self.paint_label
        patch = self.sess.run(self.stratgan.G, feed_dict={self.stratgan.z: z, 
                                                 self.stratgan.y: paint_label,
                                                 self.stratgan.is_training: False})
        r_patch = patch[0].reshape(self.config.h_dim, self.config.h_dim)
        return r_patch





    def get_core_error(self, next_patch):

        core_loc_match = np.logical_and(self.core_loc >= self.patch_xcoord_i,
                                   self.core_loc < self.patch_xcoord_i+self.patch_width-self.core_width)

        # check for anyting in the core list
        if np.any( core_loc_match ):
            
            core_idx = np.argmax(core_loc_match)

            canvas_overlap = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height,
                                         self.core_loc[core_idx]:self.core_loc[core_idx]+self.core_width]
            patch_overlap = next_patch[:, self.core_loc[core_idx]-self.patch_xcoord_i:self.core_loc[core_idx]-self.patch_xcoord_i+self.core_width]

            ec = np.linalg.norm( (patch_overlap) - (canvas_overlap))

            self.core_threshold_error = np.sqrt(  (canvas_overlap.size - np.sum(canvas_overlap)) * 0.6 ) * (1+self.patch_loop/10000)
            if self.core_threshold_error == 0.0:
                self.core_threshold_error = 6.0

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