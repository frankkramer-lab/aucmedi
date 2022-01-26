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
#              REFERENCE IMPLEMENTATION:              #
# Author: Raphael Meudec                              #
# GitHub: https://gist.github.com/RaphaelMeudec       #
# Date: Jul 18, 2019                                  #
# https://gist.github.com/RaphaelMeudec/7985b0c5eb720a#
# 29021d52b0a0be549a                                  #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External Libraries
import numpy as np
import tensorflow as tf
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                Occlusion Sensitivity                #
#-----------------------------------------------------#
class OcclusionSensitivity(XAImethod_Base):
    """ Initialization function for creating a Occlusion Sensitivity Map as XAI Method object.
    Normally, this class is used internally in the xai_decoder function in the AUCMEDI XAI module.

    This class provides functionality for running the compute_heatmap function,
    which computes a Occlusion Sensitivity Map for an image with a model.

    Args:
        model (Keras Model):               Keras model object.
        layerName (String):                Not required in Occlusion Sensitivity Maps, but defined by Abstract Base Class.
    """
    def __init__(self, model, layerName=None, patch_size=4):
        # Cache class parameters
        self.model = model
        self.patch_size = patch_size

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    """ Core function for computing the Saliency Map for a provided image and for specific classification outcome.
    The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

    Be aware that the image has to be provided in batch format.

    Args:
        image (NumPy Array):                Image matrix encoded as NumPy Array (provided as one-element batch).
        class_index (Integer):              Classification index for which the heatmap should be computed.
        eps (Float):                        Epsilon for rounding.
    """
    def compute_heatmap(self, image, class_index, eps=1e-8):
        # Utilize only image matrix instead of batch
        image = image[0]
        # Create empty sensitivity map
        sensitivity_map = np.zeros((image.shape[0], image.shape[1]))
        # Iterate the patch over the image
        for top_left_x in range(0, image.shape[0], self.patch_size):
            for top_left_y in range(0, image.shape[1], self.patch_size):
                patch = apply_grey_patch(image, top_left_x, top_left_y,
                                         self.patch_size)
                prediction = self.model.predict(np.array([patch]))[0]
                confidence = prediction[class_index]

                # Save confidence for this specific patch in the map
                sensitivity_map[
                    top_left_y:top_left_y + self.patch_size,
                    top_left_x:top_left_x + self.patch_size,
                ] = confidence
        # Return the resulting sensitivity map (automatically a heatmap)
        return sensitivity_map

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    """ Replace a part of the image with a grey patch.

    Args:
        image (numpy.ndarray):      Input image
        top_left_x (int):           Top Left X position of the applied box
        top_left_y (int):           Top Left Y position of the applied box
        patch_size (int):           Size of patch to apply

    Returns:
        numpy.ndarray:              Patched image
    """
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = 127.5
    return patched_image
