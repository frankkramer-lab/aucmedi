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
# External Libraries
import numpy as np
import tensorflow as tf
import tensorflow.math as m
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
        # Gradient model construction
        self.gradModel = tf.keras.models.Model(inputs=[self.model.inputs],
                         outputs=[self.model.get_layer(self.layerName).output,
                                  self.model.output])

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
            if len(layer.output_shape) >= 4:
                return layer.name
        # Otherwise, throw exception
        raise ValueError("Could not find 4D or 5D layer. Cannot apply Grad-CAM.")

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Grad-CAM heatmap for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as batch).
            class_index (int or list):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        Returns:
            heatmap (numpy.ndarray):            Computed Grad-CAM for provided image.
        """
        # Compute gradient for desired class index
        class_index = tf.convert_to_tensor(class_index, dtype=tf.int32)
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_out, preds) = self.gradModel(inputs)
            loss = tf.gather(preds, class_index, axis = 1)
        grads = tape.gradient(loss, conv_out)
        
        pooled_grads = tf.reduce_mean(grads, keepdims = True, axis=tf.range(1, tf.rank(grads) - 1))
        # Normalize gradients via "importance"
        heatmap = m.reduce_sum(m.multiply(conv_out, pooled_grads), axis = -1).numpy()

        # Intensity normalization to [0,1]
        min_val = np.amin(heatmap, keepdims = True, axis = tuple(range(1, len(heatmap.shape))))
        max_val = np.amax(heatmap, keepdims = True, axis = tuple(range(1, len(heatmap.shape))))
        numer = heatmap - min_val
        denom = (max_val - min_val) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
