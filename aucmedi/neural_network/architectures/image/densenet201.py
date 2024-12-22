#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
""" The classification variant of the DenseNet201 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.DenseNet201"           |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |

???+ abstract "Reference - Implementation"
    [https://keras.io/applications/#densenet](https://keras.io/applications/#densenet) <br>

???+ abstract "Reference - Publication"
    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. 25 Aug 2016.
    Densely Connected Convolutional Networks.
    <br>
    [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# Python Standard Library

# Third Party Libraries
from tensorflow.keras.applications import DenseNet201 as BaseModel

# Internal Libraries
from aucmedi.neural_network.architectures import Architecture_Base


#-----------------------------------------------------#
#           Architecture class: DenseNet201           #
#-----------------------------------------------------#
class DenseNet201(Architecture_Base):
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
    def create_model(self):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights:
            model_weights = "imagenet"
        else:
            model_weights = None

        # Obtain DenseNet201 as base model
        base_model = BaseModel(include_top=False, weights=model_weights,
                               input_tensor=None, input_shape=self.input,
                               pooling=None)
        top_model = base_model.output

        # Add classification head
        model = self.classifier.build(model_input=base_model.input,
                                      model_output=top_model)

        # Return created model
        return model
