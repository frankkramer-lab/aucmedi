#==============================================================================#
#  Author:       Fabian Wehr                                                   #
#  Copyright:    2026 IT-Infrastructure for Translational Medical Research,    #
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
# -----------------------------------------------------#
#                    Documentation                    #
# -----------------------------------------------------#
"""The classification variant of the ConvNeXt Large architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ConvNeXtLarge"         |
| Input_shape              | (384, 384)                 |
| Standardization          | "torch"                   |

Recommended alternative `Input_shape` is 224x224 pixels.

!!! warning
     ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range.
     Standardization is applied inside the architecture.

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_large.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_large.html) <br>

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
import torch
from torch import nn
from torchvision.models import convnext_large as TorchvisionModel
from torchvision.models import ConvNeXt_Large_Weights

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#          Architecture class: ConvNeXtLarge           #
# -----------------------------------------------------#
class ConvNeXtLarge(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization                #
    # ---------------------------------------------#
    def __init__(
        self,
        channels,
        input_resolution=(224, 224),
        pretrained_weights=False,
    ):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    # ---------------------------------------------#
    #         Architecture Attributes              #
    # ---------------------------------------------#

    def get_output_shape(self):
        # ConvNeXt Large has a fixed 32x downsampling ratio
        # Output channels are always 1536 for the large model
        h_out = self.input_shape[0] // 32
        w_out = self.input_shape[1] // 32
        return (h_out, w_out, 1536)

    def get_preprocess(self):
        # https://docs.pytorch.org/vision/stable/models.html
        # Return the weights transforms which include all preprocessing
        weights = ConvNeXt_Large_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                  #
    # ---------------------------------------------#
    def rechannel_first_layer(self, model):
        # If input channels differ from 3, replace the first convolutional layer.
        if self.channels == 3:
            return model

        first_conv = model[0][0]  # Access the first convolutional layer

        if first_conv is None:
            return model

        new_conv = nn.Conv2d(
            self.channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None),
        )
        with torch.no_grad():
            orig_w = first_conv.weight.data
            avg = orig_w.mean(dim=1, keepdim=True)
            new_conv.weight.data = avg.repeat(1, self.channels, 1, 1)
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        model[0][0] = new_conv
        return model

    def create_model(self):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights:
            model_weights = "DEFAULT"
        else:
            model_weights = None

        # Obtain base model (omit classification head)
        full_model = TorchvisionModel(weights=model_weights)
        base_model = full_model.features
        if self.channels != 3:
            base_model = self.rechannel_first_layer(base_model)
        return base_model
