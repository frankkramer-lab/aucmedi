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
"""The classification variant of the ResNeXt101 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ResNeXt101"            |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |

Choose pretrained weights via the torchvision `ResNeXt101_32X8D_Weights` enum and use
the `get_preprocess()` helper to obtain the correct preprocessing transforms.

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnext101_32x8d.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnext101_32x8d.html) <br>

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
import torch
from torchvision.models import resnext101_32x8d as TorchvisionModel
from torchvision.models import ResNeXt101_32X8D_Weights
import torch.nn as nn

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#           Architecture class: ResNeXt101            #
# -----------------------------------------------------#
class ResNeXt101(Architecture_Base):
    def __init__(self, channels, input_resolution=(224, 224), pretrained_weights=False):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    # ---------------------------------------------#
    #         Architecture Attributes             #
    # ---------------------------------------------#

    def get_output_shape(self):
        # Hybrid approach: fast-path for common sizes, otherwise compute by
        # running a non-pretrained forward pass and cache the result.
        if hasattr(self, "_cached_output_shape") and self._cached_output_shape:
            return self._cached_output_shape

        common = {(224, 224): (7, 7, 2048)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        full_model = TorchvisionModel(weights=None)
        modules = list(full_model.children())[:-2]
        base_model = torch.nn.Sequential(*modules)
        if self.channels != 3:
            base_model = self.rechannel_first_layer(base_model)
        base_model = base_model.cpu()
        base_model.eval()
        with torch.no_grad():
            x = torch.zeros(1, self.channels, self.input_shape[0], self.input_shape[1])
            out = base_model(x)

        if isinstance(out, dict):
            out = next(v for v in out.values() if hasattr(v, "ndim"))

        h_out = int(out.shape[2])
        w_out = int(out.shape[3])
        c_out = int(out.shape[1])
        self._cached_output_shape = (h_out, w_out, c_out)
        return self._cached_output_shape

    def get_preprocess(self):
        weights = ResNeXt101_32X8D_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def rechannel_first_layer(self, model):
        # If input channels differ from 3, replace the first convolutional layer.
        if self.channels == 3:
            return model

        first_conv = model[0]  # Access the first convolutional layer

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

        model[0] = new_conv
        return model

    def create_model(self):
        if self.pretrained_weights:
            weights_arg = ResNeXt101_32X8D_Weights.DEFAULT
        else:
            weights_arg = None

        full_model = TorchvisionModel(weights=weights_arg)
        base_model = nn.Sequential(*(list(full_model.children())[:-2]))
        if self.channels != 3:
            base_model = self.rechannel_first_layer(base_model)
        return base_model
