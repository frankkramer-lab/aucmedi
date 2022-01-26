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
# External libraries
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#         Abstract Base Class for XAI Methods         #
#-----------------------------------------------------#
""" An abstract base class for a XAI Method class.

Methods:
    __init__                Object creation function.
    compute_heatmap:        Application of the XAI Method on an image.
"""
class XAImethod_Base(ABC):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating a XAI Method object.
    Normally, this class is used internally in the xai_decoder function in the AUCMEDI XAI module.

    This class provides functionality for running the compute_heatmap function,
    which computes a heatmap for an image with a model.

    Args:
        model (Keras Model):               Keras model object.
        layerName (String):                Layer name of the convolutional layer for heatmap computation.
    """
    @abstractmethod
    def __init__(self, model, layerName=None):
        pass

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    """ Core function for computing the XAI heatmap for a provided image and for specific classification outcome.

    The shape of the returned heatmap is 2D -> batch and channel axis will be removed.
    The returned heatmap should be encoded with a range of [0,1]

    Be aware that the image has to be provided in batch format.

    Args:
        image (NumPy Array):                Image matrix encoded as NumPy Array (provided as one-element batch).
        class_index (Integer):              Classification index for which the heatmap should be computed.
        eps (Float):                        Epsilon for rounding.
    """
    def compute_heatmap(self, image, class_index):
        pass
