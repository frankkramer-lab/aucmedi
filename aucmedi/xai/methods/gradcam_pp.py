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
import tensorflow as tf

# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base


#-----------------------------------------------------#
#                XAI Method: Grad-Cam++               #
#-----------------------------------------------------#
class GradCAMpp(XAImethod_Base):
    """ XAI Method for Grad-CAM++.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Author: Samson Woof <br>
        GitHub Profile: [https://github.com/samson6460](https://github.com/samson6460) <br>
        Date: May 21, 2020 <br>
        [https://github.com/samson6460/tf_keras_gradcamplusplus](https://github.com/samson6460/tf_keras_gradcamplusplus)
        <br>

    ??? abstract "Reference - Publication"
        Aditya Chattopadhay; Anirban Sarkar; Prantik Howlader; Vineeth N Balasubramanian. 07 May 2018.
        Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks.
        <br>
        [https://ieeexplore.ieee.org/document/8354201](https://ieeexplore.ieee.org/document/8354201)

    This class provides functionality for running the compute_heatmap function,
    which computes a Grad-CAM++ heatmap for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a Grad-CAM++ as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            layerName (str):                   Layer name of the convolutional layer for heatmap computation.
        """
        # Cache class parameters
        self.model = model
        self.layerName = layerName
        # Try to find output layer if not defined
        if self.layerName is None: self.layerName = self.find_output_layer()

    #---------------------------------------------#
    #            Identify Output Layer            #
    #---------------------------------------------#
    def find_output_layer(self):
        """ Internal function. Applied if `layerName==None`.

        Identify last/final convolutional layer in neural network architecture.
        This layer is used to obtain activation outputs / feature map.
        """
        # Iterate over all layers
        for layer in reversed(self.model.layers):
            # Check to see if the layer has a 4D output -> Return layer
            if len(layer.output.shape) == 4:
                return layer.name
        # Otherwise, throw exception
        raise ValueError("Could not find 4D layer. Cannot apply Grad-CAM++.")

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Grad-CAM++ heatmap for a provided image and for specific classification
        outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Grad-CAM++ for provided image.
        """
        # Gradient model construction
        gradModel = tf.keras.models.Model(inputs=[self.model.inputs],
                         outputs=[self.model.get_layer(self.layerName).output,
                                  self.model.output])

        # Compute gradient for desierd class index
        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    inputs = tf.cast(image, tf.float32)
                    (conv_output, preds) = gradModel(inputs)
                    output = preds[:, class_index]
                    conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)
        global_sum = np.sum(conv_output, axis=(0, 1, 2))

        # Normalize constants
        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, eps)
        alphas = alpha_num / alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=(0,1))
        alphas /= alpha_normalization_constant

        # Deep Linearization weighting
        weights = np.maximum(conv_first_grad[0], 0.0)
        deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
        heatmap = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
