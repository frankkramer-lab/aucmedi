#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
from tensorflow.keras.applications import imagenet_utils
import numpy as np
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#           Subfunction class: Standardize            #
#-----------------------------------------------------#
""" A Standardization method which utilizes custom normalization functions and the Keras
    preprocess_input() functionality in order to normalize intensity value ranges to be
    suitable for neural networks.

    Default mode: "z-score"
    Possible modes: ["z-score", "minmax", "grayscale", "tf", "caffe", "torch"]


Mode Descriptons:
    Custom Implementations:
    z-score:    Sample-wise Z-score normalization (also called Z-transformation).
    minmax:     Sample-wise scaling to range [0,1].
    grayscale:  Sample-wise scaling to grayscale range [0, 255].

    Keras Implementations: https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input
    caffe:      Will convert the images from RGB to BGR, then will zero-center each color channel
                with respect to the ImageNet dataset, without scaling. (RGB encoding required!)
    tf:         Will scale pixels between -1 and 1, sample-wise. (Grayscale/RGB encoding required!)
    torch:      Will scale pixels between 0 and 1 and then will normalize each channel with respect
                to the ImageNet dataset.  (RGB encoding required!)

Methods:
    __init__                Object creation function
    transform:              Standardize an image input according to selected mode.
"""
class Standardize(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score", smooth=0.000001):
        # Verify mode existence
        if mode not in ["z-score", "minmax", "grayscale", "tf", "caffe", "torch"]:
            raise ValueError("Subfunction - Standardize: Unknown modus", mode)
        # Cache class variables
        self.mode = mode
        self.e = smooth

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform z-score normalization
        if self.mode == "z-score":
            # Compute mean and standard deviation
            mean = np.mean(image)
            std = np.std(image)
            # Scaling
            image_norm = (image - mean + self.e) / (std  + self.e)
        # Perform MinMax normalization between [0,1]
        elif self.mode == "minmax":
            # Identify minimum and maximum
            max_value = np.max(image)
            min_value = np.min(image)
            # Scaling
            image_norm = (image - min_value + self.e) / \
                         (max_value - min_value + self.e)
        elif self.mode == "grayscale":
            # Identify minimum and maximum
            max_value = np.max(image)
            min_value = np.min(image)
            # Scaling
            image_scaled = (image - min_value + self.e) / \
                           (max_value - min_value + self.e)
            image_norm = np.around(image_scaled * 255, decimals=0)
        else:
            # Verify if image is in [0,255] format
            if np.min(image) < 0 or np.max(image) > 255:
                raise ValueError("Subfunction Standardize: Image values are not in range [0,255]!",
                    "Provided min/max values for image are:", np.min(image), np.max(image),
                    "Ensure that all images are normalized to [0,255] before using the following modes:",
                    "['tf', 'caffe', 'torch']")
            # Perform architecture standardization
            image_norm = imagenet_utils.preprocess_input(image, mode=self.mode)
        # Return standardized image
        return image_norm
