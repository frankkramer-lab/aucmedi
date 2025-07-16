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
# External Libraries
import numpy as np
import tensorflow as tf
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#     Gradient-weighted Class Activation Mapping      #
#-----------------------------------------------------#
class GradCAM(XAImethod_Base):
    """ XAI Method for Gradient-weighted Class Activation Mapping (Grad-CAM).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: François Chollet <br>
        Date: April 26, 2020 <br>
        [https://keras.io/examples/vision/grad_cam/](https://keras.io/examples/vision/grad_cam/) <br>

    ??? abstract "Reference - Implementation #2"
        Author: Adrian Rosebrock <br>
        Date: March 9, 2020 <br>
        [https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/) <br>

    ??? abstract "Reference - Publication"
        Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. 7 Oct 2016.
        Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
        <br>
        [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

    This class provides functionality for running the compute_heatmap function,
    which computes a Grad-CAM heatmap for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a Grad-CAM as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            layerName (str):                   Layer name of the convolutional layer for heatmap computation.
        """
        # Cache class parameters
        self.model = model
        self.layerName = layerName
        # Try to find output layer if not defined
        if self.layerName is None : self.layerName = self.find_output_layer()

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
            if len(layer.output.shape) >= 4:
                return layer.name
        # Otherwise, throw exception
        raise ValueError("Could not find 4D layer. Cannot apply Grad-CAM.")

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Grad-CAM heatmap for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D or 3D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Grad-CAM for provided image.
        """
        # Gradient model construction
        layer_output = self.model.get_layer(self.layerName).output
        model_output = self.model.output
        if isinstance(model_output, list):
            outputs = [layer_output] + model_output
        else:
            outputs = [layer_output, model_output]

        gradModel = tf.keras.models.Model(inputs=self.model.inputs,
                         outputs=outputs)
        # Compute gradient for desired class index
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_out, preds) = gradModel(inputs)
            loss = preds[:, class_index]
        grads = tape.gradient(loss, conv_out)
        # Identify pooling axis
        if len(image.shape) == 4 : pooling_axis = (0, 1, 2)
        else : pooling_axis = (0, 1, 2, 3)
        # Averaged output gradient based on feature map of last conv layer
        pooled_grads = tf.reduce_mean(grads, axis=pooling_axis)
        # Normalize gradients via "importance"
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
