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
# Author: Aakash Kumar Nain                           #
# GitHub: https://github.com/AakashKumarNain          #
# Date: Jun 02, 2020                                  #
#https://keras.io/examples/vision/integrated_gradients#
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                     04 Mar 2017.                    #
#       Axiomatic Attribution for Deep Networks.      #
#      Mukund Sundararajan, Ankur Taly, Qiqi Yan.     #
#           https://arxiv.org/abs/1703.01365          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External Libraries
import numpy as np
import tensorflow as tf
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                 Integrated Gradients                #
#-----------------------------------------------------#
class IntegratedGradients(XAImethod_Base):
    """ Initialization function for creating a Integrated Gradients Map as XAI Method object.
    Normally, this class is used internally in the xai_decoder function in the AUCMEDI XAI module.

    This class provides functionality for running the compute_heatmap function,
    which computes a Integrated Gradients Map for an image with a model.

    Args:
        model (Keras Model):               Keras model object.
        layerName (String):                Not required in Integrated Gradients Maps, but defined by Abstract Base Class.
        num_steps (Integer):               Number of iterations for interpolation.
    """
    def __init__(self, model, layerName=None, num_steps=50):
        # Cache class parameters
        self.model = model
        self.num_steps = num_steps

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    """ Core function for computing the Integrated Gradients Map for a provided image and for specific classification outcome.
    The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

    Be aware that the image has to be provided in batch format.

    Args:
        image (NumPy Array):                Image matrix encoded as NumPy Array (provided as one-element batch).
        class_index (Integer):              Classification index for which the heatmap should be computed.
        eps (Float):                        Epsilon for rounding.
    """
    def compute_heatmap(self, image, class_index, eps=1e-8):
        # Perform interpolation
        baseline = np.zeros(image.shape).astype(np.float32)
        interpolated_imgs = []
        for step in range(0, self.num_steps + 1):
            cii = baseline + (step / self.num_steps) * (image - baseline)
            interpolated_imgs.append(cii)
        interpolated_imgs = np.array(interpolated_imgs).astype(np.float32)

        # Get the gradients for each interpolated image
        grads = []
        for int_img in interpolated_imgs:
            # Compute gradient
            with tf.GradientTape() as tape:
                inputs = tf.cast(int_img, tf.float32)
                tape.watch(inputs)
                preds = self.model(inputs)
                loss = preds[:, class_index]
            gradient = tape.gradient(loss, inputs)
            # Add to gradient list
            grads.append(gradient[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        # Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)
        # Calculate integrated gradients
        integrated_grads = (image - baseline) * avg_grads
        # Obtain maximum gradient
        integrated_grads = tf.reduce_max(integrated_grads, axis=-1)

        # Convert to NumPy & Remove batch axis
        heatmap = integrated_grads.numpy()[0,:,:]
        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
