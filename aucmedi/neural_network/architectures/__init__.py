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
# Abstract Base Class for Architectures
from aucmedi.neural_network.architectures.arch_base import Architecture_Base

#-----------------------------------------------------#
#                    Architectures                    #
#-----------------------------------------------------#
# Vanilla Classifier
from aucmedi.neural_network.architectures.vanilla import Architecture_Vanilla
# DenseNet
from aucmedi.neural_network.architectures.densenet121 import Architecture_DenseNet121
from aucmedi.neural_network.architectures.densenet169 import Architecture_DenseNet169
# EfficientNet
from aucmedi.neural_network.architectures.efficientnetb0 import Architecture_EfficientNetB0
from aucmedi.neural_network.architectures.efficientnetb1 import Architecture_EfficientNetB1
from aucmedi.neural_network.architectures.efficientnetb2 import Architecture_EfficientNetB2
from aucmedi.neural_network.architectures.efficientnetb3 import Architecture_EfficientNetB3
from aucmedi.neural_network.architectures.efficientnetb4 import Architecture_EfficientNetB4
from aucmedi.neural_network.architectures.efficientnetb5 import Architecture_EfficientNetB5
from aucmedi.neural_network.architectures.efficientnetb6 import Architecture_EfficientNetB6
from aucmedi.neural_network.architectures.efficientnetb7 import Architecture_EfficientNetB7
# InceptionResNet
from aucmedi.neural_network.architectures.inceptionresnetv2 import Architecture_InceptionResNetV2
# InceptionV3
from aucmedi.neural_network.architectures.inceptionv3 import Architecture_InceptionV3
# ResNet
from aucmedi.neural_network.architectures.resnet50 import Architecture_ResNet50
# MobileNet
from aucmedi.neural_network.architectures.mobilenet import Architecture_MobileNet
from aucmedi.neural_network.architectures.mobilenetv2 import Architecture_MobileNetV2

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#
# Architecture Dictionary
architecture_dict = {"Vanilla": Architecture_Vanilla,
                     "ResNet50": Architecture_ResNet50,
                     "DenseNet121": Architecture_DenseNet121,
                     "DenseNet169": Architecture_DenseNet169,
                     "EfficientNetB0": Architecture_EfficientNetB0,
                     "EfficientNetB1": Architecture_EfficientNetB1,
                     "EfficientNetB2": Architecture_EfficientNetB2,
                     "EfficientNetB3": Architecture_EfficientNetB3,
                     "EfficientNetB4": Architecture_EfficientNetB4,
                     "EfficientNetB5": Architecture_EfficientNetB5,
                     "EfficientNetB6": Architecture_EfficientNetB6,
                     "EfficientNetB7": Architecture_EfficientNetB7,
                     "InceptionResNetV2": Architecture_InceptionResNetV2,
                     "InceptionV3": Architecture_InceptionV3,
                     "MobileNet": Architecture_MobileNet,
                     "MobileNetV2": Architecture_MobileNetV2
}
# List of implemented architectures
architectures = list(architecture_dict.keys())

#-----------------------------------------------------#
#       Meta Information of Architecture Classes      #
#-----------------------------------------------------#
# Utilized standardize mode of architectures required for Transfer Learning
supported_standardize_mode = {"Vanilla": "tf",
                              "ResNet50": "caffe",
                              "DenseNet121": "torch",
                              "DenseNet169": "torch",
                              "EfficientNetB0": "caffe",
                              "EfficientNetB1": "caffe",
                              "EfficientNetB2": "caffe",
                              "EfficientNetB3": "caffe",
                              "EfficientNetB4": "caffe",
                              "EfficientNetB5": "caffe",
                              "EfficientNetB6": "caffe",
                              "EfficientNetB7": "caffe",
                              "InceptionResNetV2": "tf",
                              "InceptionV3": "tf",
                              "MobileNet": "tf",
                              "MobileNetV2": "tf"
}
