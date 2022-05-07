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
""" The classification variant of the ResNet152 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ResNet152"             |
| Input_shape              | (224, 224)                 |
| Standardization          | "caffe"                    |

???+ abstract "Reference - Implementation"
    https://keras.io/api/applications/resnet/ <br>

???+ abstract "Reference - Publication"
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. 10 Dec 2015.
    Deep Residual Learning for Image Recognition.
    <br>
    https://arxiv.org/abs/1512.03385
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.applications.resnet import ResNet152
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#            Architecture class: ResNet152            #
#-----------------------------------------------------#
class Architecture_ResNet152(Architecture_Base):
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

        # Obtain ResNet152 as base model
        base_model = ResNet152(include_top=False, weights=model_weights,
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
