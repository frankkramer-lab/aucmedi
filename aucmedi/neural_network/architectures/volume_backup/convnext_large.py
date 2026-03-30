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
""" The classification variant of the ConvNeXt Large architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.ConvNeXtLarge"         |
| Input_shape              | (64, 64, 64)               |
| Standardization          | None                       |

!!! warning
     ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range.
     Standardization is applied inside the architecture.

???+ abstract "Reference - Implementation"
    Solovyev, Roman & Kalinin, Alexandr & Gabruseva, Tatiana. (2021). <br>
    3D Convolutional Neural Networks for Stalled Brain Capillary Detection. <br>
    [https://github.com/ZFTurbo/classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) <br>

???+ abstract "Reference - Publication"
    Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    10 Jan 2022. A ConvNet for the 2020s.
    <br>
    [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from classification_models_3D.tfkeras import Classifiers
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#          Architecture class: ConvNeXt Large         #
#-----------------------------------------------------#
class ConvNeXtLarge(Architecture_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, classification_head, channels, input_shape=(64, 64, 64),
                 pretrained_weights=False, preprocessing=True):
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights
        self.preprocessing = preprocessing

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights : model_weights = "imagenet"
        else : model_weights = None

        # Obtain ConvNeXtLarge as base model
        BaseModel, preprocess_input = Classifiers.get("convnext_large")
        base_model = BaseModel(include_top=False, weights=model_weights,
                               input_tensor=None, input_shape=self.input,
                               pooling=None, 
                               include_preprocessing=self.preprocessing)
        top_model = base_model.output

        # Add classification head
        model = self.classifier.build(model_input=base_model.input,
                                      model_output=top_model)

        # Return created model
        return model
