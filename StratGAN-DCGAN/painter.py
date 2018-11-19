# import cv2
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
Which did not carry a license at the time of use.
"""


class CanvasPainter(object):
    def __init__(self, stratgan,
                 paint_label=None, paint_width=1000, paint_height=None, 
                 overlap=8, threshold=10):

        self.sess = stratgan.sess
        self.stratgan = stratgan
        self.config = stratgan.config

        self.paint_samp_dir = self.stratgan.paint_samp_dir

        if not paint_label==0 and not paint_label:
            print('Label not given for painting, assuming zero for label')
            # self.paint_label = tf.one_hot(0, 6)
            self.paint_label = np.zeros((1,6))
            self.paint_label[0, 0] = 1
        else:
            # paint_label = tf.one_hot(paint_label, self.config.n_categories)
            self.paint_label = np.zeros((1,6))
            self.paint_label[0, paint_label] = 1

        self.paint_width = paint_width
        if not paint_height:
            self.paint_height = int(paint_width / 4)
        else:
            self.paint_height = paint_height

        self.overlap = overlap
        self.threshold = threshold

        self.patch_height = self.patch_width = self.config.h_dim
        self.patch_size = self.patch_height * self.patch_width

        # self.threshold_error = self.threshold * self.patch_size * self.overlap
        # self.threshold_error = self.threshold

        self.patch_count = int( (self.paint_width*self.paint_height) / (self.patch_size) ) 
        self.canvas = np.ones((self.paint_height, self.paint_width))

        # generate the list of patch coordinates
        self.patch_xcoords, self.patch_ycoords = self.calculate_patch_coords()

        # generate a random sample for the first patch and quilt into image
        self.patch_i = 0
        first_patch = self.generate_patch()

        # quilt into the first coord spot
        self.patch_coords_i = (self.patch_xcoords[self.patch_i], self.patch_ycoords[self.patch_i])
        self.quilt_patch(self.patch_coords_i, first_patch, mcb=None)
        self.patch_i += 1

        self.canvas_fig = plt.figure()



    def calculate_patch_coords(self):
        """
        calculate location for patches to begin, currently ignores mod() patches
        """
        w = np.hstack((np.array([0]), np.arange(self.patch_width-self.overlap, self.paint_width, self.patch_width-self.overlap)[:-1]))
        h = np.hstack((np.array([0]), np.arange(self.patch_height-self.overlap, self.paint_height, self.patch_height-self.overlap)[:-1]))
        xm, ym = np.meshgrid(w, h)
        x = xm.flatten()
        y = ym.flatten()
        # print("x:", x)
        # print("y:", y)
        return x, y


    def add_next_patch(self):
        """
        generate  new patch for quiliting, must pass error threshold
        """
        self.threshold_error = self.threshold
        self.patch_xcoord_i = self.patch_xcoords[self.patch_i]
        self.patch_ycoord_i = self.patch_ycoords[self.patch_i]
        self.patch_coords_i = (self.patch_xcoord_i, self.patch_ycoord_i)
        
        self.dbfig, self.dbax = plt.subplots(2, 4)

        match = False
        while not match:
            next_patch = self.generate_patch()

            patch_error, patch_error_surf = self.get_patch_error(next_patch)
            # print("patcherror:", patch_error)
            # print("thresh_error:", self.threshold_error)

            # if patch_error <= self.threshold_error:
            #     match = True
            print("thresh_error:", self.threshold_error)            
            # print("eh:", eh)
            # print("ev:", ev)
            print("patch_error:", patch_error)

            if len(patch_error.shape) > 0:
                patch_error = patch_error.sum()
                print(patch_error)

            if patch_error <= self.threshold_error:
                match = True
            else:
                self.threshold_error *= 1.05 # increase by 5% error threshold

        # then calculate the minimum cost boundary
        mcb = self.calculate_min_cost_boundary(patch_error_surf)

        # then quilt it
        self.quilt_patch(self.patch_coords_i, next_patch, mcb)


    def generate_patch(self):
        z = np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(np.float32)
        paint_label = self.paint_label
        patch = self.sess.run(self.stratgan.G, feed_dict={self.stratgan.z: z, 
                                                 self.stratgan.y: paint_label,
                                                 self.stratgan.is_training: False})
        r_patch = patch[0].reshape(self.config.h_dim, self.config.h_dim)
        return r_patch
        

    def fill_canvas(self):
        while self.patch_i < self.patch_xcoords.size: # self.patch_count:

            self.add_next_patch()

            sys.stdout.write("Progress : [%-20s] %d%% | [%d]/[%d] patches completed\n" % 
                ('='*int((self.patch_i*20/self.patch_count)), int(self.patch_i/self.patch_count*100),
                 self.patch_i, self.patch_count))
            # sys.stdout.flush()

            # samp = plt.imshow(self.canvas, cmap='gray')
            # plt.savefig(os.path.join(self.paint_samp_dir, '%03d.png'%self.patch_i), dpi=300, bbox_inches='tight')
            # plt.close()

            self.patch_i += 1


#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
    def get_patch_error(self, next_patch):

        if self.patch_xcoord_i == 0:
            # a left-side patch, only calculate horizontal
            e, e_surf = self.overlap_error_horizntl(next_patch)

        elif self.patch_ycoord_i == 0:
            # a top-side patch, only calculate vertical
            e, e_surf = self.overlap_error_vertical(next_patch)

        else:
            # a center patch, calculate both
            e[0], e_surf[0, :, :] = self.overlap_error_horizntl(next_patch)
            e[1], e_surf[1, :, :] = self.overlap_error_vertical(next_patch)

        return e, e_surf

    def overlap_error_vertical(self, next_patch):
        
        # print(self.canvas.shape)
        # print(self.patch_xcoord_i, self.patch_ycoord_i)
        
        canvas_overlap = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.patch_height, \
                                     self.patch_xcoord_i:self.patch_xcoord_i+self.overlap]
        patch_overlap = next_patch[:, 0:self.overlap]

        self.dbax[0,0].imshow(self.canvas, cmap='gray')
        # self.dbax[0,0].set_xlim(self.patch_xcoord_i-self.patch_width, self.patch_xcoord_i+self.patch_width)
        # self.dbax[0,0].set_ylim(self.patch_ycoord_i, self.patch_ycoord_i+self.patch_height*2)
        self.dbax[0,1].imshow(canvas_overlap, cmap='gray')
        self.dbax[0,2].imshow(next_patch, cmap='gray')
        self.dbax[0,3].imshow(patch_overlap, cmap='gray')
        self.dbax[1,0].imshow((canvas_overlap - patch_overlap)**2, cmap='gray')
        # plt.show()
        
        # print("canvascut:", [self.patch_ycoord_i,self.patch_ycoord_i+self.patch_height, \
        #                      self.patch_xcoord_i,self.patch_xcoord_i+self.overlap])
        # print("canvascutshape:", canvas_overlap.shape)
        # print("patchcutshape:", patch_overlap.shape)

        
        ev = np.linalg.norm(canvas_overlap - patch_overlap)
        ev_surf = (canvas_overlap - patch_overlap)**2
        # print(ev)
        # eh = 0.
        # sse = ev.sum()

        # fig,ax = plt.subplots(1)
        # ax.imshow(self.canvas, cmap='gray')
        # rect = patches.Rectangle((self.patch_xcoord_i, self.patch_ycoord_i), self.overlap,self.patch_height,linewidth=1,edgecolor='r',facecolor='none')
        # ax.add_patch(rect)
        # plt.savefig(os.path.join(self.paint_samp_dir, '%03d.png'%self.patch_i), dpi=300, bbox_inches='tight')
        # plt.close()

        # print("eshape:",e.shape)
        # print("sseshape:",sse.shape)

        return ev, ev_surf


    def overlap_error_horizntl(self, next_patch):

        canvas_overlap = self.canvas[self.patch_ycoord_i:self.patch_ycoord_i+self.overlap, 
                                     self.patch_xcoord_i:self.patch_xcoord_i+self.patch_width]
        patch_overlap = next_patch[0:self.overlap, :]
        
        # print("canvascut:", [self.patch_ycoord_i,self.patch_ycoord_i+self.overlap, 
        #                      self.patch_xcoord_i,self.patch_xcoord_i+self.patch_width])
        # print("canvascutshape:", canvas_overlap.shape)
        # print("patchcutshape:", patch_overlap.shape)

        eh = np.linalg.norm(canvas_overlap - patch_overlap)
        eh_surf = (canvas_overlap - patch_overlap)**2
        # ev = 0.
        # sse = eh.sum()
        
        # print("eshape:",e.shape)
        # print("sseshape:",sse.shape)
        
        return eh, eh_surf


    def OverlapErrorVertical_old( imgPx, samplePx ):
        iLeft,jLeft = imgPx
        iRight,jRight = samplePx
        OverlapErr = 0
        diff = np.zeros((3))
        for i in range( PatchSize ):
            for j in range( OverlapWidth ):
                diff[0] =  int(img[i + iLeft, j+ jLeft][0]) - int(img_sample[i + iRight, j + jRight][0])
                diff[1] =  int(img[i + iLeft, j+ jLeft][1]) - int(img_sample[i + iRight, j + jRight][1])
                diff[2] =  int(img[i + iLeft, j+ jLeft][2]) - int(img_sample[i + iRight, j + jRight][2])
                OverlapErr += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
        return OverlapErr


    def OverlapErrorHorizntl_old( leftPx, rightPx ):
        iLeft,jLeft = leftPx
        iRight,jRight = rightPx
        OverlapErr = 0
        diff = np.zeros((3))
        for i in range( OverlapWidth ):
            for j in range( PatchSize ):
                diff[0] =  int(img[i + iLeft, j+ jLeft][0]) - int(img_sample[i + iRight, j + jRight][0])
                diff[1] =  int(img[i + iLeft, j+ jLeft][1]) - int(img_sample[i + iRight, j + jRight][1])
                diff[2] =  int(img[i + iLeft, j+ jLeft][2]) - int(img_sample[i + iRight, j + jRight][2])
                OverlapErr += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
        return OverlapErr


    def GetBestPatches_old( px ):#Will get called in GrowImage
        PixelList = []
        #check for top layer
        if px[0] == 0:
            for i in range(sample_height - PatchSize):
                for j in range(OverlapWidth, sample_width - PatchSize ):
                    error = OverlapErrorVertical( (px[0], px[1] - OverlapWidth), (i, j - OverlapWidth)  )
                    if error  < ThresholdOverlapError:
                        PixelList.append((i,j))
                    elif error < ThresholdOverlapError/2:
                        return [(i,j)]
        #check for leftmost layer
        elif px[1] == 0:
            for i in range(OverlapWidth, sample_height - PatchSize ):
                for j in range(sample_width - PatchSize):
                    error = OverlapErrorHorizntl( (px[0] - OverlapWidth, px[1]), (i - OverlapWidth, j)  )
                    if error  < ThresholdOverlapError:
                        PixelList.append((i,j))
                    elif error < ThresholdOverlapError/2:
                        return [(i,j)]
        #for pixel placed inside 
        else:
            for i in range(OverlapWidth, sample_height - PatchSize):
                for j in range(OverlapWidth, sample_width - PatchSize):
                    error_Vertical   = OverlapErrorVertical( (px[0], px[1] - OverlapWidth), (i,j - OverlapWidth)  )
                    error_Horizntl   = OverlapErrorHorizntl( (px[0] - OverlapWidth, px[1]), (i - OverlapWidth,j) )
                    if error_Vertical  < ThresholdOverlapError and error_Horizntl < ThresholdOverlapError:
                        PixelList.append((i,j))
                    elif error_Vertical < ThresholdOverlapError/2 and error_Horizntl < ThresholdOverlapError/2:
                        return [(i,j)]
        return PixelList

#-----------------------------------------------------------------------------------------------#
#|                              Quilting and related Functions                                 |#
#-----------------------------------------------------------------------------------------------#

    def SSD_Error_old( offset, imgPx, samplePx ):
        err_r = int(img[imgPx[0] + offset[0], imgPx[1] + offset[1]][0]) -int(img_sample[samplePx[0] + offset[0], samplePx[1] + offset[1]][0])
        err_g = int(img[imgPx[0] + offset[0], imgPx[1] + offset[1]][1]) - int(img_sample[samplePx[0] + offset[0], samplePx[1] + offset[1]][1])
        err_b = int(img[imgPx[0] + offset[0], imgPx[1] + offset[1]][2]) - int(img_sample[samplePx[0] + offset[0], samplePx[1] + offset[1]][2])
        return (err_r**2 + err_g**2 + err_b**2)/3.0

    def SSD_Error( canvas_overlapped, patch_overlapped ):
        assert canvas_overlapped.shape == patch_overlapped.shape
        err = img[imgPx[0] + offset[0], imgPx[1] + offset[1]][0] - img_sample[samplePx[0] + offset[0], samplePx[1] + offset[1]]

        return err*err

#---------------------------------------------------------------#
#|                  Calculating Cost                           |#
#---------------------------------------------------------------#

    def calculate_min_cost_boundary(self, patch_error_surf):

        if self.patch_xcoord_i == 0:
            # a left-side patch, only calculate horizontal
            mcb = self.min_cost_path_horizntl(patch_error_surf)

        elif self.patch_ycoord_i == 0:
            # a top-side patch, only calculate vertical
            mcb = self.min_cost_path_vertical(patch_error_surf)

        else:
            # a center patch, calculate both
            mcb[0, :, :] = self.min_cost_path_horizntl(patch_error_surf[0, :, :])
            mcb[1, :, :] = self.min_cost_path_vertical(patch_error_surf[1, :, :])

        return mcb
    

    def GetCostVertical(imgPx, samplePx):
        Cost = np.zeros((PatchSize, OverlapWidth))
        for j in range(OverlapWidth):
            for i in range(PatchSize):
                if i == PatchSize - 1:
                    Cost[i,j] = SSD_Error((i ,j - OverlapWidth), imgPx, samplePx)
                else:
                    if j == 0 :
                        Cost[i,j] = SSD_Error((i , j - OverlapWidth), imgPx, samplePx) + min( SSD_Error((i + 1, j - OverlapWidth), imgPx, samplePx),SSD_Error((i + 1,j + 1 - OverlapWidth), imgPx, samplePx) )
                    elif j == OverlapWidth - 1:
                        Cost[i,j] = SSD_Error((i, j - OverlapWidth), imgPx, samplePx) + min( SSD_Error((i + 1, j - OverlapWidth), imgPx, samplePx), SSD_Error((i + 1, j - 1 - OverlapWidth), imgPx, samplePx) )
                    else:
                        Cost[i,j] = SSD_Error((i, j -OverlapWidth), imgPx, samplePx) + min(SSD_Error((i + 1, j - OverlapWidth), imgPx, samplePx),SSD_Error((i + 1, j + 1 - OverlapWidth), imgPx, samplePx), SSD_Error((i + 1, j - 1 - OverlapWidth), imgPx, samplePx))
        return Cost

    def GetCostHorizntl(imgPx, samplePx):
        Cost = np.zeros((OverlapWidth, PatchSize))
        for i in range( OverlapWidth ):
            for j in range( PatchSize ):
                if j == PatchSize - 1:
                    Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx)
                elif i == 0:
                    Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error((i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i + 1 - OverlapWidth, j + 1), imgPx, samplePx))
                elif i == OverlapWidth - 1:
                    Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error((i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i - 1 - OverlapWidth, j + 1), imgPx, samplePx))
                else:
                    Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error((i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i + 1 - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i - 1 - OverlapWidth, j + 1), imgPx, samplePx))
        return Cost

    #---------------------------------------------------------------#
    #|                  Finding Minimum Cost Path                  |#
    #---------------------------------------------------------------#

    def min_cost_path_vertical(self, patch_error_surf):
        Boundary = np.zeros((self.patch_height), np.int)
        ParentMatrix = np.zeros((self.patch_height, self.overlap), np.int)
        # print("BoundaryShape:", Boundary.shape)
        # print("ParentMatrixShape:", ParentMatrix.shape)
        # print(patch_error_surf)
        for i in np.arange(1, self.patch_height):
            # for the height of the patch
            for j in np.arange(self.overlap):
                # for each col in overlap
                if j == 0:
                    # if first col in row
                    # print("i:", i, "j:", j)
                    ParentMatrix[i,j] = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j+1] else j+1
                elif j == self.overlap - 1:
                    # if last col in row
                    ParentMatrix[i,j] = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j-1] else j-1
                else:
                    # if center cols
                    curr_min = j if patch_error_surf[i-1,j] < patch_error_surf[i-1,j-1] else j-1
                    ParentMatrix[i,j] = curr_min if patch_error_surf[i-1,curr_min] < patch_error_surf[i-1,j+1] else j+1
                
                patch_error_surf[i,j] += patch_error_surf[i-1, ParentMatrix[i,j]]
        

        self.dbax[1,1].imshow(ParentMatrix, cmap='gray')
        # plt.show()


        minIndex = 0
        for j in np.arange(1, self.overlap):
            minIndex = minIndex if patch_error_surf[self.patch_height - 1, minIndex] < patch_error_surf[self.patch_height - 1, j] else j
        
        Boundary[self.patch_height-1] = minIndex
        for i in np.arange(self.patch_height - 1, 0, -1):
            Boundary[i - 1] = ParentMatrix[i, Boundary[i]]

        self.dbax[1,1].plot(Boundary, np.arange(self.patch_height), 'r')
        print(self.patch_xcoord_i + Boundary)
        self.dbax[0,0].plot(self.patch_xcoord_i + Boundary, np.arange(self.patch_height), 'r')

        # plt.show()
        
        return Boundary


    def min_cost_path_horizntl(self, patch_error_surf):
        Boundary = np.zeros((self.patch_width), np.int)
        ParentMatrix = np.zeros((self.overlap, self.patch_width), np.int)
        for j in np.arange(1, self.patch_width):
            for i in np.arange(self.overlap):
                if i == 0:
                    ParentMatrix[i,j] = i if patch_error_surf[i,j-1] < patch_error_surf[i+1,j-1] else i + 1
                
                elif i == self.overlap - 1:
                    ParentMatrix[i,j] = i if patch_error_surf[i,j-1] < patch_error_surf[i-1,j-1] else i - 1
                
                else:
                    curr_min = i if patch_error_surf[i,j-1] < patch_error_surf[i-1,j-1] else i - 1
                    ParentMatrix[i,j] = curr_min if patch_error_surf[curr_min,j-1] < patch_error_surf[i-1,j-1] else i + 1
                
                patch_error_surf[i,j] += patch_error_surf[ParentMatrix[i,j], j-1]

        minIndex = 0
        for i in np.arange(1,self.overlap):
            minIndex = minIndex if patch_error_surf[minIndex, self.patch_width - 1] < patch_error_surf[i, self.patch_width - 1] else i
        
        Boundary[self.patch_width-1] = minIndex
        for j in np.arange(self.patch_width - 1,0,-1):
            Boundary[j - 1] = ParentMatrix[Boundary[j],j]
        
        return Boundary





    def FindMinCostPathVertical_old(Cost):
        Boundary = np.zeros((PatchSize),np.int)
        ParentMatrix = np.zeros((PatchSize, OverlapWidth),np.int)
        for i in range(1, PatchSize):
            for j in range(OverlapWidth):
                if j == 0:
                    ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j+1] else j+1
                elif j == OverlapWidth - 1:
                    ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
                else:
                    curr_min = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
                    ParentMatrix[i,j] = curr_min if Cost[i-1,curr_min] < Cost[i-1,j+1] else j+1
                Cost[i,j] += Cost[i-1, ParentMatrix[i,j]]
        minIndex = 0
        for j in range(1,OverlapWidth):
            minIndex = minIndex if Cost[PatchSize - 1, minIndex] < Cost[PatchSize - 1, j] else j
        Boundary[PatchSize-1] = minIndex
        for i in range(PatchSize - 1,0,-1):
            Boundary[i - 1] = ParentMatrix[i,Boundary[i]]
        return Boundary

    def FindMinCostPathHorizntl(Cost):
        Boundary = np.zeros(( PatchSize),np.int)
        ParentMatrix = np.zeros((OverlapWidth, PatchSize),np.int)
        for j in range(1, PatchSize):
            for i in range(OverlapWidth):
                if i == 0:
                    ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i+1,j-1] else i + 1
                elif i == OverlapWidth - 1:
                    ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
                else:
                    curr_min = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
                    ParentMatrix[i,j] = curr_min if Cost[curr_min,j-1] < Cost[i-1,j-1] else i + 1
                Cost[i,j] += Cost[ParentMatrix[i,j], j-1]
        minIndex = 0
        for i in range(1,OverlapWidth):
            minIndex = minIndex if Cost[minIndex, PatchSize - 1] < Cost[i, PatchSize - 1] else i
        Boundary[PatchSize-1] = minIndex
        for j in range(PatchSize - 1,0,-1):
            Boundary[j - 1] = ParentMatrix[Boundary[j],j]
        return Boundary

#---------------------------------------------------------------#
#|                      Quilting                               |#
#---------------------------------------------------------------#

    def quilt_patch(self, coords, patch, mcb=None):

        if mcb is None:
            print("NO MCB")
            y = coords[0]
            x = coords[1]
            self.canvas[x:x+self.patch_height, y:y+self.patch_width] = np.squeeze(patch)
        else:

            if self.patch_xcoord_i == 0:
                # a left-side patch, only calculate horizontal
                self.quilt_patch_horizntl(coords, patch, mcb)

            elif self.patch_ycoord_i == 0:
                # a top-side patch, only calculate vertical
                self.quilt_patch_vertical(coords, patch, mcb)

            else:
                # a center patch, calculate both
                self.quilt_patch_horizntl(coords, patch, mcb[0, :, :])
                self.quilt_patch_vertical(coords, patch, mcb[1, :, :])

            # self.canvas[x:x+self.patch_height, y:y+self.patch_width] = np.squeeze(patch)      

    def quilt_patch_vertical(self, coords, patch, mcb):
        y = coords[0]
        x = coords[1]
        print(mcb)
        print(y, x)
        for i in np.arange(self.patch_height):
            for j in np.arange(mcb[i], self.patch_width):
                # print(j)
                self.canvas[x+i, y+j] = patch[i, j]

        self.dbax[0,0].imshow(self.canvas, cmap='gray')
        self.dbax[0,0].plot(self.patch_xcoord_i + mcb, np.arange(self.patch_height), 'r')
        plt.show()


    def quilt_patch_horizntl(self, coords, patch, mcb):
        y = coords[0]
        x = coords[1]
        for i in np.arange(self.patch_width):
            for j in np.arange(mcb[i], self.patch_height):
                self.canvas[x+i, y+j] = patch[i, j]
            

    def QuiltVertical(Boundary, imgPx, samplePx):
        for i in range(PatchSize):
            for j in range(Boundary[i], 0, -1):
                img[imgPx[0] + i, imgPx[1] - j] = img_sample[ samplePx[0] + i, samplePx[1] - j ]

    def QuiltHorizntl(Boundary, imgPx, samplePx):
        for j in range(PatchSize):
            for i in range(Boundary[j], 0, -1):
                img[imgPx[0] - i, imgPx[1] + j] = img_sample[samplePx[0] - i, samplePx[1] + j]

    def QuiltPatches( imgPx, samplePx ):
        #check for top layer
        if imgPx[0] == 0:
            Cost = GetCostVertical(imgPx, samplePx)
            # Getting boundary to stitch
            Boundary = FindMinCostPathVertical(Cost)
            #Quilting Patches
            QuiltVertical(Boundary, imgPx, samplePx)
        #check for leftmost layer
        elif imgPx[1] == 0:
            Cost = GetCostHorizntl(imgPx, samplePx)
            #Boundary to stitch
            Boundary = FindMinCostPathHorizntl(Cost)
            #Quilting Patches
            QuiltHorizntl(Boundary, imgPx, samplePx)
        #for pixel placed inside 
        else:
            CostVertical = GetCostVertical(imgPx, samplePx)
            CostHorizntl = GetCostHorizntl(imgPx, samplePx)
            BoundaryVertical = FindMinCostPathVertical(CostVertical)
            BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
            QuiltVertical(BoundaryVertical, imgPx, samplePx)
            QuiltHorizntl(BoundaryHorizntl, imgPx, samplePx)

#--------------------------------------------------------------------------------------------------------#
#                                   Growing Image Patch-by-patch                                        |#
#--------------------------------------------------------------------------------------------------------#

    def FillImage( imgPx, samplePx ):
        for i in range(PatchSize):
            for j in range(PatchSize):
                img[ imgPx[0] + i, imgPx[1] + j ] = img_sample[ samplePx[0] + i, samplePx[1] + j ]


def THE_MAIN_ROUTINE():
    pixelsCompleted = 0
    TotalPatches = ( (img_height - 1 )/ PatchSize )*((img_width)/ PatchSize) - 1

    print(TotalPatches)
    print(pixelsCompleted)

    # sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: --.------" % ('='*(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted))
    sys.stdout.write("Progress : [{patchcmplt}] / [{TotalPatches}]".format(patchcmplt=pixelsCompleted, TotalPatches=TotalPatches))
    sys.stdout.flush()
    it = 0
    while GrowPatchLocation[0] + PatchSize < img_height:
        pixelsCompleted += 1
        # print(pixelsCompleted)
        ThresholdConstant = InitialThresConstant
        #set progress to zer0
        progress = 0 
        while progress == 0:
            ThresholdOverlapError = ThresholdConstant * PatchSize * OverlapWidth
            #Get Best matches for current pixel
            print(GrowPatchLocation)
            List = GetBestPatches(GrowPatchLocation)
            if len(List) > 0:
                progress = 1
                #Make A random selection from best fit pxls
                sampleMatch = List[ randint(0, len(List) - 1) ]
                FillImage( GrowPatchLocation, sampleMatch )
                #Quilt this with in curr location
                QuiltPatches( GrowPatchLocation, sampleMatch )
                #upadate cur pixel location
                GrowPatchLocation = (GrowPatchLocation[0], GrowPatchLocation[1] + PatchSize)
                if GrowPatchLocation[1] + PatchSize > img_width:
                    GrowPatchLocation = (GrowPatchLocation[0] + PatchSize, 0)
            #if not progressed, increse threshold
            else:
                ThresholdConstant *= 1.1
            it += 1
            samp = plt.imshow(img)
            plt.savefig('samp/samp_%05d.png' % it, bbox_inches='tight')
            plt.close()

        # print pixelsCompleted, ThresholdConstant
        sys.stdout.write('\r')
        # sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: %f" % ('='*(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted, ThresholdConstant))
        sys.stdout.write("Progress : [{patchcmplt}] / [{TotalPatches}]".format(patchcmplt=pixelsCompleted, TotalPatches=TotalPatches))
        sys.stdout.flush()
        
    # Displaying Images

    samp = plt.imshow(img_sample)
    plt.savefig('sample.png', bbox_inches='tight')
    plt.close()

    gen = plt.imshow(img)
    plt.savefig('gener.png', bbox_inches='tight')
    plt.close()

