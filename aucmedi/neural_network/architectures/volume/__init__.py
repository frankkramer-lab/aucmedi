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
# Abstract Base Class for Architectures
from aucmedi.neural_network.architectures.arch_base import Architecture_Base

#-----------------------------------------------------#
#                    Architectures                    #
#-----------------------------------------------------#
# Vanilla Classifier
from aucmedi.neural_network.architectures.volume.vanilla import Architecture_Vanilla
# DenseNet
from aucmedi.neural_network.architectures.volume.densenet121 import Architecture_DenseNet121
from aucmedi.neural_network.architectures.volume.densenet169 import Architecture_DenseNet169
from aucmedi.neural_network.architectures.volume.densenet201 import Architecture_DenseNet201
# ResNet
from aucmedi.neural_network.architectures.volume.resnet18 import Architecture_ResNet18
from aucmedi.neural_network.architectures.volume.resnet34 import Architecture_ResNet34
from aucmedi.neural_network.architectures.volume.resnet50 import Architecture_ResNet50
from aucmedi.neural_network.architectures.volume.resnet101 import Architecture_ResNet101
from aucmedi.neural_network.architectures.volume.resnet152 import Architecture_ResNet152
# ResNeXt
from aucmedi.neural_network.architectures.volume.resnext50 import Architecture_ResNeXt50
from aucmedi.neural_network.architectures.volume.resnext101 import Architecture_ResNeXt101
# MobileNet
from aucmedi.neural_network.architectures.volume.mobilenet import Architecture_MobileNet
from aucmedi.neural_network.architectures.volume.mobilenetv2 import Architecture_MobileNetV2
# VGG
from aucmedi.neural_network.architectures.volume.vgg16 import Architecture_VGG16
from aucmedi.neural_network.architectures.volume.vgg19 import Architecture_VGG19

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#
# Architecture Dictionary
architecture_dict = {
    "Vanilla": Architecture_Vanilla,
    "DenseNet121": Architecture_DenseNet121,
    "DenseNet169": Architecture_DenseNet169,
    "DenseNet201": Architecture_DenseNet201,
    "ResNet18": Architecture_ResNet18,
    "ResNet34": Architecture_ResNet34,
    "ResNet50": Architecture_ResNet50,
    "ResNet101": Architecture_ResNet101,
    "ResNet152": Architecture_ResNet152,
    "ResNeXt50": Architecture_ResNeXt50,
    "ResNeXt101": Architecture_ResNeXt101,
    "MobileNet": Architecture_MobileNet,
    "MobileNetV2": Architecture_MobileNetV2,
    "VGG16": Architecture_VGG16,
    "VGG19": Architecture_VGG19,
}
# List of implemented architectures
architectures = list(architecture_dict.keys())

#-----------------------------------------------------#
#       Meta Information of Architecture Classes      #
#-----------------------------------------------------#
# Utilized standardize mode of architectures required for Transfer Learning
supported_standardize_mode = {
    "Vanilla": "z-score",
    "DenseNet121": "torch",
    "DenseNet169": "torch",
    "DenseNet201": "torch",
    "ResNet18": "grayscale",
    "ResNet34": "grayscale",
    "ResNet50": "grayscale",
    "ResNet101": "grayscale",
    "ResNet152": "grayscale",
    "ResNeXt50": "grayscale",
    "ResNeXt101": "grayscale",
    "MobileNet": "tf",
    "MobileNetV2": "tf",
    "VGG16": "caffe",
    "VGG19": "caffe",
}
