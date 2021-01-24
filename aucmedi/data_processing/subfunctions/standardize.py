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
from tensorflow.keras.applications import imagenet_utils
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#           Subfunction class: Standardize            #
#-----------------------------------------------------#
""" A Standardization method which utilizes the Keras preprocess_input() functionality
    in order to normalize intensity value ranges to be suitable for neural networks.

    Default mode: "tf"
    Possible modes: ["tf", "caffe", "torch"]

    Source: https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input
    caffe: will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
    tf: will scale pixels between -1 and 1, sample-wise.
    torch: will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset.

Methods:
    __init__                Object creation function
    transform:              Standardize an image input according to selected mode.
"""
class Standardize(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="tf"):
        self.mode = mode

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform architecture standardization
        image_norm = imagenet_utils.preprocess_input(image, mode=self.mode)
        # Return standardized image
        return image_norm
