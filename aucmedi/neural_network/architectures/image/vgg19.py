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
"""The classification variant of the VGG19 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.VGG19"                 |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |

Choose pretrained weights via the torchvision `VGG19_Weights` enum and use
the `get_preprocess()` helper to obtain the correct preprocessing transforms.

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html) <br>

???+ abstract "Reference - Publication"
    Karen Simonyan, Andrew Zisserman. 04 Sep 2014.
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    <br>
    [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
"""
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from torchvision.models import vgg19 as BaseModel
from torchvision.models import VGG19_Weights

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#              Architecture class: VGG19              #
# -----------------------------------------------------#
class VGG19(Architecture_Base):
    def __init__(self, channels, input_resolution=(224, 224), pretrained_weights=False):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    def get_output_shape(self):
        # VGG final conv layer has 512 channels and 32x downsampling
        # Hybrid: fast-path for common size, otherwise derive shape by a
        # non-pretrained forward pass and cache the result.
        if hasattr(self, "_cached_output_shape") and self._cached_output_shape:
            return self._cached_output_shape

        common = {(224, 224): (7, 7, 512)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        import torch

        full_model = BaseModel(weights=None)
        # VGG features are available as .features
        base_model = getattr(full_model, "features", None)
        if base_model is None:
            modules = list(full_model.children())[:-1]
            base_model = torch.nn.Sequential(*modules)
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
        weights = VGG19_Weights.DEFAULT
        return weights.transforms()

    def create_model(self):
        if self.pretrained_weights:
            weights_arg = VGG19_Weights.DEFAULT
        else:
            weights_arg = None

        full_model = BaseModel(weights=weights_arg)
        base_model = full_model.features
        return base_model
