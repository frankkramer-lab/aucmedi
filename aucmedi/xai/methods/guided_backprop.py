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
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                Guided Backpropagation               #
#-----------------------------------------------------#
class GuidedBackpropagation(XAImethod_Base):
    """ XAI Method for Guided Backpropagation.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: Hoa Nguyen <br>
        GitHub Profile: https://nguyenhoa93.github.io/ <br>
        Date: Jul 29, 2020 <br>
        https://stackoverflow.com/questions/55924331/how-to-apply-guided-backprop-in-tensorflow-2-0 <br>

    ??? abstract "Reference - Implementation #2"
        Author: Huynh Ngoc Anh <br>
        GitHub Profile: https://github.com/experiencor <br>
        Date: Jun 23, 2017 <br>
        https://github.com/experiencor/deep-viz-keras/ <br>

    ??? abstract "Reference - Implementation #3"
        Author: Tim <br>
        Date: Jan 25, 2019 <br>
        https://stackoverflow.com/questions/54366935/make-a-deep-copy-of-a-keras-model-in-python <br>

    ??? abstract "Reference - Publication"
        Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. 21 Dec 2014.
        Striving for Simplicity: The All Convolutional Net.
        <br>
        https://arxiv.org/abs/1412.6806

    This class provides functionality for running the compute_heatmap function,
    which computes a Guided Backpropagation for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating Guided Backpropagation as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            layerName (str):                   Not required in Guided Backpropagation, but defined by Abstract Base Class.
        """
        # Create a deep copy of the model
        model_copy = tf.keras.models.clone_model(model)
        model_copy.build(model.input.shape)
        model_copy.compile(optimizer=model.optimizer, loss=model.loss)
        model_copy.set_weights(model.get_weights())

        # Define custom Relu activation function
        @tf.custom_gradient
        def guidedRelu(x):
            def grad(dy):
                return tf.cast(dy>0, "float32") * tf.cast(x>0, "float32") * dy
            return tf.nn.relu(x), grad
        # Replace Relu activation layers with custom Relu activation layer
        layer_dict = [layer for layer in model_copy.layers if hasattr(layer, "activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        # Cache class parameters
        self.model = model_copy

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Guided Backpropagation for a provided image and for specific classification outcome.

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
            heatmap (numpy.ndarray):            Computed Guided Backpropagation for provided image.
        """
        # Compute gradient for desierd class index
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            preds = self.model(inputs)
            loss = preds[:, class_index]
        gradient = tape.gradient(loss, inputs)
        # Obtain maximum gradient based on feature map of last conv layer
        gradient = tf.reduce_max(gradient, axis=-1)
        # Convert to NumPy & Remove batch axis
        heatmap = gradient.numpy()[0,:,:]

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
