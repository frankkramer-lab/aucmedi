#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
# Python Standard Library

# Third Party Libraries
import numpy as np
from tensorflow.keras.applications import imagenet_utils

# Internal Libraries
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base


#-----------------------------------------------------#
#           Subfunction class: Standardize            #
#-----------------------------------------------------#
class Standardize(Subfunction_Base):
    """ A Standardization method which utilizes custom normalization functions and the Keras
        preprocess_input() functionality in order to normalize intensity value ranges to be
        suitable for neural networks.

    Default mode: `"z-score"`

    Possible modes: `["z-score", "minmax", "grayscale", "tf", "caffe", "torch"]`


    ???+ info "Mode Descriptions"

        | Mode                | Description                                                               |
        | ------------------- | ------------------------------------------------------------------------- |
        | `"z-score"`         | Sample-wise Z-score normalization (also called Z-transformation).         |
        | `"minmax"`          | Sample-wise scaling to range [0,1].                                       |
        | `"grayscale"`       | Sample-wise scaling to grayscale range [0, 255].                          |
        | `"caffe"`           | Will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling. (RGB encoding required!) |
        | `"tf"`              | Will scale pixels between -1 and 1, sample-wise. (Grayscale/RGB encoding required!) |
        | `"torch"`           | Will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset. (RGB encoding required!) |

    ??? abstract "Reference - Implementation"
        Keras preprocess_input() for `"tf", "caffe", "torch"`

        https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input
    """ # noqa E501
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score", per_channel=False, smooth=0.000001):
        """ Initialization function for creating a Standardize Subfunction which can be passed to a
            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            mode (str):             Selected mode which standardization/normalization technique should be applied.
            per_channel (bool):     Option to apply standardization per channel instead of across complete image.
            smooth (float):         Smoothing factor to avoid zero devisions (epsilon).
        """
        # Verify mode existence
        if mode not in ["z-score", "minmax", "grayscale", "tf", "caffe", "torch"]:
            raise ValueError("Subfunction - Standardize: Unknown modus", mode)
        # Cache class variables
        self.mode = mode
        self.per_channel = per_channel
        self.e = smooth

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Apply normalization per channel
        if self.per_channel:
            image_norm = image.copy()
            for c in range(0, image.shape[-1]):
                image_norm[..., c] = self.normalize(image[..., c])
        # Apply normalization across complete image
        else:
            image_norm = self.normalize(image)
        # Return standardized image
        return image_norm

    #---------------------------------------------#
    #      Internal Function: Normalization       #
    #---------------------------------------------#
    def normalize(self, image):
        # Perform z-score normalization
        if self.mode == "z-score":
            # Compute mean and standard deviation
            mean = np.mean(image)
            std = np.std(image)
            # Scaling
            image_norm = (image - mean + self.e) / (std + self.e)
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
        # Return normalized image
        return image_norm
