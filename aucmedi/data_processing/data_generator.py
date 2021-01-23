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
            samples (List of Strings):
            path_imagedir (String):         Path to the directory containing the images.
            labels (NumPy Array):           Path to the index/class annotation file if required. (csv/json)
            data_aug (DataAugmentation):    Boolean option whether annotation data is available.
            subfunctions (List of Subfunctions):
                                            Boolean option whether annotation data is sparse categorical or one-hot encoded.
            batch_size (Integer):           Additional parameters for the format interfaces.
            shuffle (Boolean):              Additional parameters for the format interfaces.
            grayscale (Boolean):            Additional parameters for the format interfaces.
            prepare_images (Boolean):       Additional parameters for the format interfaces.
            sample_weights (List of Floats):Additional parameters for the format interfaces.
            seed (Integer):                 Additional parameters for the format interfaces.
    """
    def __init__(self, samples, path_imagedir, labels=None, data_aug=None,
                 subfunctions=[], batch_size=32, shuffle=False, grayscale=False,
                 prepare_images=False, sample_weights=None, seed=None):
        # Return the
        super(ImageIterator, self).__init__(len(samples), batch_size, shuffle, seed)



    def _get_batches_of_transformed_samples(self, index_array):
        return None

    # for loop over index_array list:
    # -> load image np.array(Image.open(filepath))
    # -> data augmentation
    # -> get or run subfunctions on dataset (possible caching)
    # return batch(img, class, weight)


# multiprocessing: for loop via map
