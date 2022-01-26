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
#             REFERENCE IMPLEMENTATION #1:            #
# Author: François Chollet                            #
# Date: April 26, 2020                                #
# https://keras.io/examples/vision/grad_cam/          #
#-----------------------------------------------------#
#             REFERENCE IMPLEMENTATION #2:            #
# Author: Adrian Rosebrock                            #
# Date: March 9, 2020                                 #
# https://www.pyimagesearch.com/2020/03/09/grad-cam-v #
# isualize-class-activation-maps-with-keras-tensorflo #
# w-and-deep-learning/                                #
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                     7 Oct 2016.                     #
#   Grad-CAM: Visual Explanations from Deep Networks  #
#           via Gradient-based Localization.          #
#     Ramprasaath R. Selvaraju, Michael Cogswell,     #
#   Abhishek Das, Ramakrishna Vedantam, Devi Parikh,  #
#                     Dhruv Batra.                    #
#           https://arxiv.org/abs/1610.02391          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External Libraries
import numpy as np
import tensorflow as tf
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#     Gradient-weighted Class Activation Mapping      #
#-----------------------------------------------------#
class GradCAM(XAImethod_Base):
    """ Initialization function for creating a Grad-Cam as XAI Method object.
    Normally, this class is used internally in the xai_decoder function in the AUCMEDI XAI module.

    This class provides functionality for running the compute_heatmap function,
    which computes a Grad-Cam heatmap for an image with a model.

    Args:
        model (Keras Model):               Keras model object.
        layerName (String):                Layer name of the convolutional layer for heatmap computation.
    """
    def __init__(self, model, layerName=None):
        # Cache class parameters
        self.model = model
        self.layerName = layerName
        # Try to find output layer if not defined
        if self.layerName is None : self.layerName = self.find_output_layer()

    #---------------------------------------------#
    #            Identify Output Layer            #
    #---------------------------------------------#
    """ Identify last/final convolutional layer in neural network architecture.
        This layer is used to obtain activation outputs / feature map.
    """
    def find_output_layer(self):
        # Iterate over all layers
        for layer in reversed(self.model.layers):
            # Check to see if the layer has a 4D output -> Return layer
            if len(layer.output_shape) == 4:
                return layer.name
        # Otherwise, throw exception
        raise ValueError("Could not find 4D layer. Cannot apply Grad-CAM.")

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    """ Core function for computing the Grad-Cam heatmap for a provided image and for specific classification outcome.
    The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

    Be aware that the image has to be provided in batch format.

    Args:
        image (NumPy Array):                Image matrix encoded as NumPy Array (provided as one-element batch).
        class_index (Integer):              Classification index for which the heatmap should be computed.
        eps (Float):                        Epsilon for rounding.
    """
    def compute_heatmap(self, image, class_index, eps=1e-8):
        # Gradient model construction
        gradModel = tf.keras.models.Model(inputs=[self.model.inputs],
                         outputs=[self.model.get_layer(self.layerName).output,
                                  self.model.output])
        # Compute gradient for desierd class index
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_out, preds) = gradModel(inputs)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, conv_out)
        # Averaged output gradient based on feature map of last conv layer
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # Normalize gradients via "importance"
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
