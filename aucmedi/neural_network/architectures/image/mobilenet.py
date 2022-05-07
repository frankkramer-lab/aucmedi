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
""" The classification variant of the MobileNet architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.MobileNet"             |
| Input_shape              | (224, 224)                 |
| Standardization          | "tf"                       |

???+ abstract "Reference - Implementation"
    https://keras.io/api/applications/mobilenet/ <br>

???+ abstract "Reference - Publication"
    Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand,
    Marco Andreetto, Hartwig Adam. 17 Apr 2017.
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
    <br>
    https://arxiv.org/abs/1704.04861
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.applications import MobileNet
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#            Architecture class: MobileNet            #
#-----------------------------------------------------#
class Architecture_MobileNet(Architecture_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, classification_head, channels, input_shape=(224, 224),
                 pretrained_weights=False):
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self, n_labels, fcl_dropout=True, activation_output="softmax",
                     pretrained_weights=False):
        # Get pretrained image weights from imagenet if desired
        if pretrained_weights : model_weights = "imagenet"
        else : model_weights = None

        # Obtain MobileNet as base model
        base_model = MobileNet(include_top=False, weights=model_weights,
                              input_tensor=None, input_shape=self.input,
                              pooling=None)
        top_model = base_model.output

        # Add classification head as top model
        top_model = layers.GlobalAveragePooling2D(name="avg_pool")(top_model)
        if fcl_dropout:
            top_model = layers.Dense(units=512)(top_model)
            top_model = layers.Dropout(0.3)(top_model)
        top_model = layers.Dense(n_labels, name="preds")(top_model)
        top_model = layers.Activation(activation_output, name="probs")(top_model)

        # Create model
        model = Model(inputs=base_model.input, outputs=top_model)

        # Return created model
        return model
