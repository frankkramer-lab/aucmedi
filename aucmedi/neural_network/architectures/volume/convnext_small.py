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
"""The classification variant of the ConvNeXt Small architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.ConvNeXtSmall"         |
| Input_shape              | (64, 64, 64, channels)     |
| Standardization          | torch                      |

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
import torch

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#         Architecture class: ConvNeXt Small          #
# -----------------------------------------------------#
class ConvNeXtSmall(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(
        self,
        channels=3,
        input_resolution=(64, 64, 64),
        pretrained_weights=False,
        preprocessing=True,
    ):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.preprocessing = preprocessing
        self.channels = channels

    # ---------------------------------------------#
    #         Architecture Attributes              #
    # ---------------------------------------------#

    def get_output_shape(self):
        # ConvNeXt Small has a fixed 32x downsampling ratio
        # Output channels are typically 768 for the small model
        h_out = self.input_shape[0] // 32
        w_out = self.input_shape[1] // 32
        d_out = self.input_shape[2] // 32
        return (h_out, w_out, d_out, 768)

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def rechannel_first_layer(self, model):
        # If input channels differ from 3, replace the first convolutional layer.
        if self.channels == 3:
            return model
        first_conv = model[0][0]  # Access the first convolutional layer

        if first_conv is None:
            return model

        # Use the same convolution class (Conv2d/Conv3d) as the original layer
        conv_cls = first_conv.__class__

        new_conv = conv_cls(
            self.channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None),
        )

        with torch.no_grad():
            orig_w = first_conv.weight.data
            # average over input channel dimension to initialize new input channels
            avg = orig_w.mean(dim=1, keepdim=True)
            # avg shape: (out_channels, 1, k1, k2[, k3])
            # repeat to match new number of input channels
            repeat_dims = (1, self.channels, 1, 1, 1)
            new_conv.weight.data = avg.repeat(*repeat_dims)
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        model[0][0] = new_conv
        return model

    def create_model(self):
        full_model = create_model(
            "convnext_small",
            pretrained=self.pretrained_weights,
            num_classes=0,
            global_pool="",
        )
        base_model = nn.Sequential(*list(full_model.children())[:-1])

        if self.channels != 3:
            base_model = self.rechannel_first_layer(base_model)
        return base_model
