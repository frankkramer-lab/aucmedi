# ==============================================================================#
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
# ==============================================================================#
# -----------------------------------------------------#
#                    Documentation                    #
# -----------------------------------------------------#
"""The classification variant of the ConvNeXt Base architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.ConvNeXtBase"          |
| Input_shape              | (64, 64, 64, 3)            |
| Standardization          | None                       |

!!! warning
     ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range.
     Standardization is applied inside the architecture.

???+ abstract "Reference - Implementation"
    Solovyev. (2022). <br>
    3D Convolutional Neural Networks for Stalled Brain Capillary Detection. <br>
    [https://github.com/ZFTurbo/timm_3d](https://github.com/ZFTurbo/timm_3d) <br>

???+ abstract "Reference - Publication"
    Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    10 Jan 2022. A ConvNet for the 2020s.
    <br>
    [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)
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
#          Architecture class: ConvNeXt Base          #
# -----------------------------------------------------#
class ConvNeXtBase(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(
        self,
        channels,
        input_resolution=(64, 64, 64),
        pretrained_weights=False,
    ):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights

    def get_output_shape(self):
        # ConvNeXt Base has a fixed 32x downsampling ratio
        # Output channels are always 1024 for the base model
        h_out = self.input_shape[0] // 32
        w_out = self.input_shape[1] // 32
        d_out = self.input_shape[2] // 32
        return (h_out, w_out, d_out, 1024)

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def create_model(self):
        full_model = create_model(
            "convnext_base",
            pretrained=self.pretrained_weights,
            num_classes=0,  # Exclude the classification head
            global_pool="",
        )
        base_model = nn.Sequential(
            *list(full_model.children())[:-1]
        )  # Exclude the final classification head

        # Return created model
        return base_model
