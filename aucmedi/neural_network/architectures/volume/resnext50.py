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
"""The classification variant of the ResNeXt50 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.ResNeXt50"             |
| Input_shape              | (64, 64, 64)               |
| Standardization          | "grayscale"                |

???+ abstract "Reference - Implementation"
    Solovyev, Roman & Kalinin, Alexandr & Gabruseva, Tatiana. (2021). <br>
    3D Convolutional Neural Networks for Stalled Brain Capillary Detection. <br>
    [https://github.com/ZFTurbo/timm_3d](https://github.com/ZFTurbo/timm_3d) <br>

???+ abstract "Reference - Publication"
    Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. 16 Nov 2016.
    Aggregated Residual Transformations for Deep Neural Networks.
    <br>
    [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
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
#            Architecture class: ResNeXt50            #
# -----------------------------------------------------#
class ResNeXt50(Architecture_Base):
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

    def get_output_shape(self):
        # ResNeXt50 has a fixed 32x downsampling ratio
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
        # Create model with pretrained weights if desired
        full_model = create_model(
            "resnext50_32x4d",
            pretrained=self.pretrained_weights,
            in_chans=self.input_shape[-1],
            num_classes=0,  # Exclude the classification head
            global_pool="",
        )
        # Remove classification head
        model = nn.Sequential(*list(full_model.children())[:-1])
        # Return created model
        return model
