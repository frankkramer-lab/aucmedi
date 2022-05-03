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
#     Abstract Interface for an Architecture class    #
#-----------------------------------------------------#
class Architecture_Base(ABC):
    """ An abstract base class for an Architecture class.

    This class provides functionality for running the create_model function,
    which returns a [tensorflow.keras model](https://www.tensorflow.org/api_docs/python/tf/keras/Model).

    ???+ example "Create a custom Architecture"
        ```python
        from aucmedi.neural_network.architectures import Architecture_Base

        class My_custom_Architecture(Architecture_Base):
            def __init__(self, channels, input_shape=(224, 224)):
                pass

            def create_model(self, n_labels, fcl_dropout=True, activation_output="softmax",
                             pretrained_weights=False):
                return my_keras_model
        ```

    ???+ info "Required Functions"
        | Function            | Description                                    |
        | ------------------- | ---------------------------------------------- |
        | `__init__()`        | Object creation function.                      |
        | `create_model()`    | Creating and returning the architecture model. |

    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    @abstractmethod
    def __init__(self, channels, input_shape=(224, 224)):
        """ Functions which will be called during the Architecture object creation.

        This function can be used to pass variables and options in the Architecture instance.
        The are no other mandatory required parameters for the initialization except for
        the number of channels and the input shape (x, y) for an image architecture and
        (x, y, z) for a volume architecture.

        Args:
            channels (int):                 Number of channels. For example: Grayscale->1 or RGB->3.
            input_shape (tuple):            Input shape of the image data for the first model layer (excluding channel axis).
        """
        self.input = input_shape + (channels,)

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    @abstractmethod
    def create_model(self, n_labels, fcl_dropout=True, activation_output="softmax",
                     pretrained_weights=False):
        """ Create the deep learning or convolutional neural network model.

        This function will be called inside the AUCMEDI model class and have to return a functional
        Keras model. The model itself should be created here or in a subfunction called
        by this function.

        Modi for activation_output: Check out [TensorFlow.Keras doc on activation functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations).

        The fully connected layer and dropout option utilizes a 512 unit Dense layer with 30% Dropout.

        Args:
            n_labels (int):                 Number of classes/labels (important for the last layer of classification head).
            fcl_dropout (bool):             Option whether to utilize a Dense & Dropout layer in the last classification layer.
            activation_output (str):           Activation function which should be used in the last classification layer.
            pretrained_weights (bool):      Option whether to utilize pretrained weights e.g. for ImageNet.

        Returns:
            model (Keras model):            A Keras model.
        """
        return None
