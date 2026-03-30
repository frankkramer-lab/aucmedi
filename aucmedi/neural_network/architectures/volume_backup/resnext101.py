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
""" The classification variant of the ResNeXt101 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.ResNeXt101"            |
| Input_shape              | (64, 64, 64)               |
| Standardization          | "grayscale"                |

???+ abstract "Reference - Implementation"
    Solovyev, Roman & Kalinin, Alexandr & Gabruseva, Tatiana. (2021). <br>
    3D Convolutional Neural Networks for Stalled Brain Capillary Detection. <br>
    [https://github.com/ZFTurbo/classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) <br>

???+ abstract "Reference - Publication"
    Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. 16 Nov 2016.
    Aggregated Residual Transformations for Deep Neural Networks.
    <br>
    [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from classification_models_3D.tfkeras import Classifiers
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#           Architecture class: ResNeXt101            #
#-----------------------------------------------------#
class ResNeXt101(Architecture_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, classification_head, channels, input_shape=(64, 64, 64),
                 pretrained_weights=False):
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights : model_weights = "imagenet"
        else : model_weights = None

        # Obtain ResNeXt101 as base model
        BaseModel, preprocess_input = Classifiers.get("resnext101")
        base_model = BaseModel(include_top=False, weights=model_weights,
                               input_tensor=None, input_shape=self.input,
                               pooling=None)
        top_model = base_model.output

        # Add classification head
        model = self.classifier.build(model_input=base_model.input,
                                      model_output=top_model)

        # Return created model
        return model
