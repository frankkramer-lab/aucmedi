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
from aucmedi.xai.methods import GuidedBackpropagation, GradCAM
from aucmedi.data_processing.subfunctions import Resize

#-----------------------------------------------------#
#                   Guided Grad-CAM                   #
#-----------------------------------------------------#
class GuidedGradCAM(XAImethod_Base):
    """ XAI Method for Guided Grad-CAM.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Author: Swapnil Ahlawat <br>
        Date: Jul 06, 2020 <br>
        https://github.com/swapnil-ahlawat/Guided-GradCAM-Keras <br>

    ??? abstract "Reference - Publication #1"
        Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. 21 Dec 2014.
        Striving for Simplicity: The All Convolutional Net.
        <br>
        https://arxiv.org/abs/1412.6806

    ??? abstract "Reference - Publication #2"
        Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. 7 Oct 2016.
        Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
        <br>
        https://arxiv.org/abs/1610.02391

    This class provides functionality for running the compute_heatmap function,
    which computes a Guided Grad-CAM heatmap for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a Guided Grad-CAM as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            layerName (str):                   Layer name of the convolutional layer for heatmap computation.
        """
        # Initialize XAI methods
        self.bp = GuidedBackpropagation(model, layerName)
        self.gc = GradCAM(model, layerName)

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Guided Grad-CAM heatmap for a provided image and for specific classification outcome.

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
            heatmap (numpy.ndarray):            Computed Guided Grad-CAM for provided image.
        """
        # Compute Guided Backpropagation
        hm_bp = self.bp.compute_heatmap(image, class_index, eps)
        # Compute Grad-CAM
        hm_gc = self.gc.compute_heatmap(image, class_index, eps)
        hm_gc = Resize(shape=image.shape[1:-1]).transform(hm_gc)
        # Combine both XAI methods
        heatmap = hm_bp * hm_gc
        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        # Return the resulting heatmap
        return heatmap
