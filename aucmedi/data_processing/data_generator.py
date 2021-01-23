#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
# Internal libraries
import aucmedi.data_processing.io_interfaces as io

#-----------------------------------------------------#
#                   Static Variables                  #
#-----------------------------------------------------#
ACCEPTABLE_IMAGE_FORMATS = ["jpeg", "jpg", "tif", "tiff", "png", "bmp", "gif"]

#-----------------------------------------------------#
#                 Keras Data Generator                #
#-----------------------------------------------------#
""" Infinite Data Generator which automatically creates batches from a list of samples.
    The created batches are model ready. This generator can be supplied directly
    to the keras model fit() function.

    The Data Generator can be used for training, validation as well as for prediction.
    It supports real-time batch generation as well as beforehand preparation of batches,
    which are then temporarly stored on disk.

    The resulting batches are created based the following pipeline:
    - Image Loading
    - Optional application of Data Augmentation
    - Optional application of Subfunctions
    - Standardize image
    - Stacking processed images to a batch

    Build on top of Keras Iterator:
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/Iterator
"""
class DataGenerator(Iterator):
    #-----------------------------------------------------#
    #                    Initialization                   #
    #-----------------------------------------------------#
    """Initialization function of the Data Generator which acts as a configuraiton hub.

        If using for prediction, the 'labels' parameter have to be None.
        Data augmentation is applied even for prediction if a DataAugmentation object is provided!

        Arguments:
            samples (List of Strings):      List of sample/index encoded as Strings.
            path_imagedir (String):         Path to the directory containing the images.
            labels (NumPy Array):           Classification list with One-Hot Encoding.
            data_aug (DataAugmentation):    Data Augmentation class instance which performs diverse data augmentation techniques.
            subfunctions (List of Subfunctions):
                                            List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
            standardize_mode (String):      Standardization modus in which image intensity values are scaled.
            batch_size (Integer):           Number of samples inside a single batch.
            shuffle (Boolean):              Boolean, whether dataset should be shuffled.
            grayscale (Boolean):            Boolean, whether images are grayscale or RGB.
            prepare_images (Boolean):       Boolean, whether all images should be prepared and backup to disk before training.
            sample_weights (List of Floats):List of weights for samples.
            seed (Integer):                 Seed to ensure reproducibility for random function.
    """
    def __init__(self, samples, path_imagedir, labels=None, batch_size=32,
                 data_aug=None, subfunctions=[], standardize_mode="tf",
                 shuffle=False, grayscale=False, prepare_images=False,
                 sample_weights=None, seed=None):
        # Cache class variables
        self.samples = samples
        self.path_imagedir = path_imagedir
        self.labels = labels
        self.data_aug = data_aug
        self.subfunctions = subfunctions
        self.grayscale = grayscale
        self.prepare_images = prepare_images
        self.sample_weights = sample_weights
        #


        # Return the
        super(ImageIterator, self).__init__(len(samples), batch_size, shuffle, seed)


    #-----------------------------------------------------#
    #              Batch Generation Function              #
    #-----------------------------------------------------#
    """adasd
    """
    def _get_batches_of_transformed_samples(self, index_array):
        # batch_x = [None] * len(index_array)

        # for i in index_array:
            # if prepared:
                # -> load prepared image
            # else:
                # -> load image
                # -> get or run subfunctions on dataset (possible caching)
                # -> data augmentation
                # -> apply standardize

        return None



    # for loop over index_array list:
    # -> load image np.array(Image.open(filepath))
    # -> data augmentation
    # -> get or run subfunctions on dataset (possible caching)
    # return batch(img, class, weight)


# multiprocessing: for loop via map
