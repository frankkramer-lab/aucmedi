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
#           Saliency Maps / Backpropagation           #
#-----------------------------------------------------#
class SaliencyMap(XAImethod_Base):
    """ XAI Method for Saliency Map (also called Backpropagation).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: Yasuhiro Kubota <br>
        GitHub Profile: https://github.com/keisen <br>
        Date: Aug 11, 2020 <br>
        https://github.com/keisen/tf-keras-vis/ <br>

    ??? abstract "Reference - Implementation #2"
        Author: Huynh Ngoc Anh <br>
        GitHub Profile: https://github.com/experiencor <br>
        Date: Jun 23, 2017 <br>
        https://github.com/experiencor/deep-viz-keras/ <br>

    ??? abstract "Reference - Publication"
        Karen Simonyan, Andrea Vedaldi, Andrew Zisserman. 20 Dec 2013.
        Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.
        <br>
        https://arxiv.org/abs/1312.6034

    This class provides functionality for running the compute_heatmap function,
    which computes a Saliency Map for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a Saliency Map as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            layerName (str):                   Not required in Saliency Maps, but defined by Abstract Base Class.
        """
        # Cache class parameters
        self.model = model

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Saliency Map for a provided image and for specific classification outcome.

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
            heatmap (numpy.ndarray):            Computed Saliency Map for provided image.
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
