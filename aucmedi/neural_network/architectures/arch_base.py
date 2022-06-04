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
        from tensorflow.keras import Input
        from tensorflow.keras.layers import Conv2D, MaxPooling2D

        class My_custom_Architecture(Architecture_Base):
            def __init__(self, classification_head, channels, input_shape=(224, 224),
                         pretrained_weights=False):
                self.classifier = classification_head
                self.input = input_shape + (channels,)
                self.pretrained_weights = pretrained_weights

            def create_model(self):
                # Initialize input layer
                model_input = Input(shape=self.input)

                # Add whatever architecture you want
                model_base = Conv2D(filters=32)(model_input)
                model_base = Conv2D(filters=64)(model_base)
                model_base = MaxPooling2D(pool_size=2)(model_base)

                # Add classification head via Classifier
                my_keras_model = self.classifier.build(model_input=model_input,
                                                      model_output=model_base)
                # Return created model
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
    def __init__(self, classification_head, channels, input_shape=(224, 224),
                 pretrained_weights=False):
        """ Functions which will be called during the Architecture object creation.

        This function can be used to pass variables and options in the Architecture instance.

        There are some mandatory required parameters for the initialization: The classification head as
        [Classifier][aucmedi.neural_network.architectures.classifier], the number of channels, and the
        input shape (x, y) for an image architecture or (x, y, z) for a volume architecture.

        Args:
            classification_head (Classifier):   Classifier object for building the classification head of the model.
            channels (int):                     Number of channels. For example: Grayscale->1 or RGB->3.
            input_shape (tuple):                Input shape of the image data for the first model layer (excluding channel axis).
            pretrained_weights (bool):          Option whether to utilize pretrained weights e.g. for ImageNet.
        """
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    @abstractmethod
    def create_model(self):
        """ Create the deep learning or convolutional neural network model.

        This function will be called inside the AUCMEDI model class and have to return a functional
        Keras model. The model itself should be created here or in a subfunction called
        by this function.

        At the end of the model building process, the classification head must be appended
        via calling the `build()` function of a [Classifier][aucmedi.neural_network.architectures.classifier].

        Returns:
            model (tf.keras model):            A Keras model.
        """
        return None
