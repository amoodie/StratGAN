"""
image serving classes for model
"""

import glob
import numpy as np
from PIL import Image
import os
import utils
import tensorflow as tf 

class BaseImageProvider(object):
    """
    This class provides a basis for generating a tf.data.Dataset and Iterator.
    It takes some hints the UNet implementation in tf.
    """
    def __init__(self, image_dir, image_ext='*.png', 
                 c_dim=None, a_min=None, a_max=None, verbose=False):

        self.verbose = verbose

        self.image_dir = image_dir
        self.image_ext = image_ext
        self._image_list = self._list_images(os.path.join(self.image_dir, self.image_ext))
        self._label_list = self._parse_labels(self._image_list)

        self.n_images = len(self._image_list)
        self.n_categories = np.unique(np.array([l for l in self._label_list])).size
        self.y_dim = self.n_categories

        assert len(self._image_list) > 0, "No training files"
        assert len(self._image_list) == len(self._label_list), "Unequal images/labels length"

        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

        # try:
        test_img = self.__load_image(self._image_list[0])
        # except:
            # Exception('could not load a test image, aborting...')

        if c_dim is None:
            print('Number of channels (c_dim) not provided, attempting to determine...')
            self.c_dim = test_img.shape[-1]
            print('Tested image had {0} dimension(s)'.format(self.c_dim))
        else:
            self.c_dim = c_dim

        self.w_dim = test_img.shape[0]
        self.h_dim = test_img.shape[1]

        if self.verbose:
            self.print_data_info()


    def _list_images(self, image_dir):
        all_files = glob.glob(image_dir)
        return [name for name in all_files]

    def _parse_labels(self, image_list):
        """
        split the labels out of the image list
        """
        path_splits = [path.split('/') for path in image_list]
        image_names = [split[-1] for split in path_splits]
        label_splits = [label.split('_') for label in image_names]
        labels = [int(split[0]) for split in label_splits]
        return labels

    def __load_image(self, path, dtype=np.float32):
        """
        single image reader, used for testing image to determine values if not given
        """
        try:
            img_array = np.array(Image.open(path), dtype)
        except:
            img_array = np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))
        return img_array

    def _make_inputs_fig(self, image_list, label_list):
        pass

    def print_data_info(self):
        print('\n')
        print('Image directory: ', self.image_dir)
        print('Number of images: %s' % self.n_images)
        print('Categories in labels: ', self.n_categories)
        print('Image height: ', self.h_dim)
        print('Image width: ', self.w_dim)
        print('\n')
        #
        #
        # add more...



class ImageDatasetProvider(BaseImageProvider):
    def __init__(self, image_dir, image_ext='*.png', c_dim=None, batch_size=None, 
                 shuffle_data=True, buffer_size=1, drop_remainder=True, repeat_data=True,
                 a_min=None, a_max=None, verbose=False):
        super().__init__(image_dir, image_ext, c_dim, a_min, a_max, verbose)
        
        self.shuffle_data = shuffle_data
        self.repeat_data = repeat_data
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder

        # convert to constants for tf
        self.filenames = tf.constant(self._image_list)
        self.labels = tf.constant(self._label_list)

        # create dataset
        self.data = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))

        # map image in the dataset
        self.data = self.data.map(self._load_image_func)

        # process options to the dataset object
        if len(self._label_list) < batch_size:
            raise Exception("dataset size is less than batch_size")
        if not self.drop_remainder:
            raise Exception("only supporting drop remainder at present")

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self._image_list) # full batch
            print('Warning: no batch size given, using full batch')

        self.data = self.data.batch(self.batch_size, drop_remainder=drop_remainder)
        self.n_batches = self.n_images // self.batch_size

        if self.shuffle_data:
            self.data = self.data.shuffle(self.buffer_size)

        if self.repeat_data:
            self.data = self.data.repeat()

        # create iterator and final input tensors
        self.iterator = self.data.make_one_shot_iterator()
        # self.image_batch, self.label_batch = self.iterator.get_next()
        self.next_batch = self.iterator.get_next()

        # self.data = self.data.prefetch(1)
        
        self.data_shape = [self.batch_size, self.h_dim, self.w_dim, self.c_dim]


    def _load_image_func(self, filename, label):
        """
        load image function used to batch the files
        """
        image_string = tf.read_file(filename)

        # decode using jpeg
        image_decoded = tf.image.decode_jpeg(image_string, channels=self.c_dim)
        
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)
        # image = tf.cast(image_decoded, tf.float32) # same as above?

        # make the label vector a one-hot
        one_hot_label = tf.one_hot(label, self.n_categories)
        
        return image, one_hot_label    


    # def _load_image_func(self, filename, label):
    #     """
    #     load image function used to batch the files
    #     """
    #     image = np.zeros((28,28,1))
    #     one_hot_label = np.array(([0, 0, 1, 0]))
        
    #     return image, one_hot_label
