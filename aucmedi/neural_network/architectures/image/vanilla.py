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
#                    Documentation                    #
#-----------------------------------------------------#
""" The classification variant of the Vanilla architecture.

No intensive hardware requirements, which makes it ideal for debugging.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.Vanilla"               |
| Input_shape              | (224, 224)                 |
| Standardization          | "z-score"                  |

???+ abstract "Reference - Implementation"
    https://github.com/wanghsinwei/isic-2019/ <br>
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#                 Vanilla Architecture                #
#-----------------------------------------------------#
class Vanilla(Architecture_Base):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, classification_head, channels, input_shape=(224, 224),
                 pretrained_weights=False):
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self):
        # Initialize input
        model_input = Input(shape=self.input)

        # Add 4x convolutional layers with increasing filters
        model_base = Conv2D(filters=32, kernel_size=3, padding='same',
                            activation='relu')(model_input)
        model_base = MaxPooling2D(pool_size=2)(model_base)

        model_base = Conv2D(filters=64, kernel_size=3, padding='same',
                            activation='relu')(model_base)
        model_base = MaxPooling2D(pool_size=2)(model_base)

        model_base = Conv2D(filters=128, kernel_size=3, padding='same',
                            activation='relu')(model_base)
        model_base = MaxPooling2D(pool_size=2)(model_base)

        model_base = Conv2D(filters=256, kernel_size=3, padding='same',
                            activation='relu')(model_base)
        model_base = MaxPooling2D(pool_size=2)(model_base)

        # Add classification head
        model = self.classifier.build(model_input=model_input,
                                      model_output=model_base)

        # Return created model
        return model
