#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
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
#     Abstract Interface for an Architecture class    #
#-----------------------------------------------------#
""" An abstract base class for a Architecture class.

Methods:
    __init__                Object creation function
    create_model:           Creating a Keras model
"""
class Architecture_Base(ABC):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    """ Functions which will be called during the Architecture object creation.
        This function can be used to pass variables and options in the Architecture instance.
        The are no mandatory required parameters for the initialization.

        Parameter:
            None
        Return:
            None
    """
    @abstractmethod
    def __init__(self):
        pass

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    """ Create the deep learning or convolutional neural network model.
        This function will be called inside the AUCMEDI model class and have to return a functional
        Keras model. The model itself should be created here or in a subfunction called
        by this function.

        Modi for out_activation: Check Keras activation function documentation.

        Parameter:
            n_labels (Integer):             Number of classes/labels (important for the last layer).
            channels (Integer):             Number of channels in the image (RGB:3, Grayscale:1).
            input_shape (Tuple):            Input shape of the image data for the first model layer.
            out_activation (String):        Activation function which should be used in the last classification layer.
            pretrained_weights (Boolean):   Option whether to utilize pretrained weights e.g. for ImageNet.
            dropout (Boolean):              Option whether to utilize a dropout layer in the last classification layer.
        Return:
            model (Keras model):        A Keras model
    """
    @abstractmethod
    def create_model(self, n_labels, channels=3, input_shape=(224, 224),
                     out_activation="softmax", pretrained_weights=False,
                     dropout=True):
        return None
