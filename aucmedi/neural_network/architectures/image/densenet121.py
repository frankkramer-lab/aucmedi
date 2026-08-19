# ==============================================================================#
#  Author:       Dominik Müller                                                #
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
# ==============================================================================#
# -----------------------------------------------------#
#                    Documentation                    #
# -----------------------------------------------------#
"""The classification variant of the DenseNet121 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.DenseNet121"           |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html) <br>

???+ abstract "Reference - Publication"
    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. 25 Aug 2016.
    Densely Connected Convolutional Networks.
    <br>
    [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
"""

# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import torch
from torch import nn
from torchvision.models import densenet121 as TorchvisionModel
from torchvision.models import DenseNet121_Weights
import torchvision.transforms as transforms_module

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#           Architecture class: DenseNet121           #
# -----------------------------------------------------#
class DenseNet121(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
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
    #         Architecture Attributes             #
    # ---------------------------------------------#

    def get_output_shape(self):
        # Hybrid: fast-path for common size, otherwise compute via non-
        # pretrained forward pass and cache the result.
        if hasattr(self, "_cached_output_shape") and self._cached_output_shape:
            return self._cached_output_shape

        common = {(224, 224): (7, 7, 1024)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        import torch

        full_model = TorchvisionModel(weights=None)
        base_model = full_model.features
        if self.channels != 3:
            base_model = self.rechannel_first_layer(base_model)
        base_model = base_model.cpu()
        base_model.eval()
        with torch.no_grad():
            x = torch.zeros(1, self.channels, self.input_shape[0], self.input_shape[1])
            out = base_model(x)

        h_out = int(out.shape[2])
        w_out = int(out.shape[3])
        c_out = int(out.shape[1])
        self._cached_output_shape = (h_out, w_out, c_out)
        return self._cached_output_shape

    def get_preprocess(self):
        weights = DenseNet121_Weights.DEFAULT
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
