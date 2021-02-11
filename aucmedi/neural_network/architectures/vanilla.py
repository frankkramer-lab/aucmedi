#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
#              REFERENCE IMPLEMENTATION:              #
#       https://github.com/wanghsinwei/isic-2019      #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Activation
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#                 Vanilla Architecture                #
#-----------------------------------------------------#
""" A vanilla image classification model.
    No intensive hardware requirements, which makes it ideal for debugging.

Methods:
    __init__                Object creation function
    create_model:           Creating a Keras model
"""
class Architecture_Vanilla(Architecture_Base):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, channels, input_shape=(224, 224)):
        self.input = input_shape + (channels,)

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self, n_labels, fcl_dropout=True, out_activation="softmax",
                     pretrained_weights=False):
        # Initialize model
        model = Sequential()

        # Add 4x convolutional layers with increasing filters
        model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                         activation='relu', input_shape=self.input))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=256, kernel_size=3, padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        # Add classification head
        model.add(GlobalAveragePooling2D())
        if fcl_dropout:
            model.add(Dense(units=512))
            model.add(Dropout(rate=0.3))
        model.add(Dense(n_labels, name="preds"))
        model.add(Activation(out_activation, name="probs"))

        # Return created model
        return model
