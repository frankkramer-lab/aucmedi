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
from aucmedi.neural_network.architectures.image.vanilla import Vanilla
# DenseNet
from aucmedi.neural_network.architectures.image.densenet121 import DenseNet121
from aucmedi.neural_network.architectures.image.densenet169 import DenseNet169
from aucmedi.neural_network.architectures.image.densenet201 import DenseNet201
# EfficientNet
from aucmedi.neural_network.architectures.image.efficientnetb0 import EfficientNetB0
from aucmedi.neural_network.architectures.image.efficientnetb1 import EfficientNetB1
from aucmedi.neural_network.architectures.image.efficientnetb2 import EfficientNetB2
from aucmedi.neural_network.architectures.image.efficientnetb3 import EfficientNetB3
from aucmedi.neural_network.architectures.image.efficientnetb4 import EfficientNetB4
from aucmedi.neural_network.architectures.image.efficientnetb5 import EfficientNetB5
from aucmedi.neural_network.architectures.image.efficientnetb6 import EfficientNetB6
from aucmedi.neural_network.architectures.image.efficientnetb7 import EfficientNetB7
# InceptionResNet
from aucmedi.neural_network.architectures.image.inceptionresnetv2 import InceptionResNetV2
# InceptionV3
from aucmedi.neural_network.architectures.image.inceptionv3 import InceptionV3
# ResNet
from aucmedi.neural_network.architectures.image.resnet50 import ResNet50
from aucmedi.neural_network.architectures.image.resnet101 import ResNet101
from aucmedi.neural_network.architectures.image.resnet152 import ResNet152
# ResNetv2
from aucmedi.neural_network.architectures.image.resnet50v2 import ResNet50V2
from aucmedi.neural_network.architectures.image.resnet101v2 import ResNet101V2
from aucmedi.neural_network.architectures.image.resnet152v2 import ResNet152V2
# ResNeXt
from aucmedi.neural_network.architectures.image.resnext50 import ResNeXt50
from aucmedi.neural_network.architectures.image.resnext101 import ResNeXt101
# MobileNet
from aucmedi.neural_network.architectures.image.mobilenet import MobileNet
from aucmedi.neural_network.architectures.image.mobilenetv2 import MobileNetV2
# NasNet
from aucmedi.neural_network.architectures.image.nasnetlarge import NASNetLarge
from aucmedi.neural_network.architectures.image.nasnetmobile import NASNetMobile
# VGG
from aucmedi.neural_network.architectures.image.vgg16 import VGG16
from aucmedi.neural_network.architectures.image.vgg19 import VGG19
# Xception
from aucmedi.neural_network.architectures.image.xception import Xception
# Vision Transformer (ViT)
from aucmedi.neural_network.architectures.image.vit_b16 import ViT_B16
from aucmedi.neural_network.architectures.image.vit_b32 import ViT_B32
from aucmedi.neural_network.architectures.image.vit_l16 import ViT_L16
from aucmedi.neural_network.architectures.image.vit_l32 import ViT_L32

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#
# Architecture Dictionary
architecture_dict = {
    "Vanilla": Vanilla,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
    "ResNeXt50": ResNeXt50,
    "ResNeXt101": ResNeXt101,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
    "InceptionResNetV2": InceptionResNetV2,
    "InceptionV3": InceptionV3,
    "MobileNet": MobileNet,
    "MobileNetV2": MobileNetV2,
    "NASNetMobile": NASNetMobile,
    "NASNetLarge": NASNetLarge,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "Xception": Xception,
    "ViT_B16": ViT_B16,
    "ViT_B32": ViT_B32,
    "ViT_L16": ViT_L16,
    "ViT_L32": ViT_L32,
}
""" Dictionary of implemented 2D Architectures Methods in AUCMEDI.

    The base key (str) or an initialized Architecture can be passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] class as `architecture` parameter.

    ???+ example "Example"
        ```python title="Recommended via NeuralNetwork class"
        my_model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.Xception",
                                  input_shape(512, 512), activation_output="softmax")
        ```

        ```python title="Manual via architecture_dict import"
        from aucmedi.neural_network.architectures import Classifier, architecture_dict

        classification_head = Classifier(n_labels=4, activation_output="softmax")
        my_arch = architecture_dict["2D.Xception"](classification_head,
                                                   channels=3, input_shape=(512,512))

        my_model = NeuralNetwork(n_labels=None, channels=None, architecture=my_arch)
        ```

        ```python title="Manual via module import"
        from aucmedi.neural_network.architectures import Classifier
        from aucmedi.neural_network.architectures.image import Xception

        classification_head = Classifier(n_labels=4, activation_output="softmax")
        my_arch = Xception(classification_head,
                                        channels=3, input_shape=(512,512))

        my_model = NeuralNetwork(n_labels=None, channels=None, architecture=my_arch)
        ```

    ???+ warning
        If passing an architecture key to the NeuralNetwork class, be aware that you have to add "2D." in front of it.

        For example:
        ```python
        # for the image architecture "ResNeXt101"
        architecture="2D.ResNeXt101"
        ```

    Architectures are based on the abstract base class [aucmedi.neural_network.architectures.arch_base.Architecture_Base][].
"""

# List of implemented architectures
architectures = list(architecture_dict.keys())

#-----------------------------------------------------#
#       Meta Information of Architecture Classes      #
#-----------------------------------------------------#
# Utilized standardize mode of architectures required for Transfer Learning
supported_standardize_mode = {
    "Vanilla": "z-score",
    "ResNet50": "caffe",
    "ResNet101": "caffe",
    "ResNet152": "caffe",
    "ResNet50V2": "tf",
    "ResNet101V2": "tf",
    "ResNet152V2": "tf",
    "ResNeXt50": "torch",
    "ResNeXt101": "torch",
    "DenseNet121": "torch",
    "DenseNet169": "torch",
    "DenseNet201": "torch",
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
    "MobileNetV2": "tf",
    "NASNetMobile": "tf",
    "NASNetLarge": "tf",
    "VGG16": "caffe",
    "VGG19": "caffe",
    "Xception": "tf",
    "ViT_B16": "tf",
    "ViT_B32": "tf",
    "ViT_L16": "tf",
    "ViT_L32": "tf",
}
""" Dictionary of recommended [Standardize][aucmedi.data_processing.subfunctions.standardize] techniques for 2D Architectures Methods in AUCMEDI.

    The base key (str) can be passed to the [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator] as `standardize_mode` parameter.

    ???+ info
        If training a new model from scratch, any Standardize technique can be used at will. <br>
        However, if training via transfer learning, it is required to use the recommended Standardize technique!

    ???+ example "Example"
        ```python title="Recommended via the NeuralNetwork class"
        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")

        my_dg = DataGenerator(samples, "images_dir/", labels=None,
                              resize=my_model.meta_input,                  # (224, 224)
                              standardize_mode=my_model.meta_standardize)  # "torch"
        ```

        ```python title="Manual via supported_standardize_mode import"
        from aucmedi.neural_network.architectures import supported_standardize_mode
        sf_norm = supported_standardize_mode["2D.DenseNet121"]

        my_dg = DataGenerator(samples, "images_dir/", labels=None,
                              resize=(224, 224),                           # (224, 224)
                              standardize_mode=sf_norm)                    # "torch"
        ```

    ???+ warning
        If using an architecture key for the supported_standardize_mode dictionary, be aware that you have to add "2D." in front of it.

        For example:
        ```python
        # for the image architecture "ResNeXt101"
        from aucmedi.neural_network.architectures import supported_standardize_mode
        sf_norm = supported_standardize_mode["2D.ResNeXt101"]
        ```
"""
