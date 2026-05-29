# ==============================================================================#
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
# ==============================================================================#
# -----------------------------------------------------#
#                    Documentation                    #
# -----------------------------------------------------#
"""The classification variant of the ResNet50 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.ResNet50"              |
| Input_shape              | (64, 64, 64)               |
| Standardization          | "torch"                |

???+ abstract "Reference - Implementation"
    Solovyev, Roman & Kalinin, Alexandr & Gabruseva, Tatiana. (2021). <br>
    3D Convolutional Neural Networks for Stalled Brain Capillary Detection. <br>
    [https://github.com/ZFTurbo/classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) <br>

???+ abstract "Reference - Publication"
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. 10 Dec 2015.
    Deep Residual Learning for Image Recognition.
    <br>
    [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
"""

# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from timm_3d import create_model
from torch import nn

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#            Architecture class: ResNet50             #
# -----------------------------------------------------#
class ResNet50(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(
        self,
        channels=3,
        input_resolution=(64, 64, 64),
        pretrained_weights=False,
    ):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    # ---------------------------------------------#
    #         Architecture Attributes              #
    # ---------------------------------------------#

    def get_output_shape(self):
        # ResNet50 has a fixed 32x downsampling ratio
        return (
            self.input_shape[0] // 32,
            self.input_shape[1] // 32,
            self.input_shape[2] // 32,
            2048,
        )

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def create_model(self):
        full_model = create_model(
            "resnet50",
            pretrained=self.pretrained_weights,
            in_chans=self.channels,
            num_classes=0,
            global_pool="",
        )
        # Remove the final pooling layer and classifier
        model = nn.Sequential(*list(full_model.children())[:-1])

        # Return created model
        return model
