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
| Key in architecture_dict | "2D.ResNet50"              |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |


Choose pretrained weights "IMAGENET1K_V1" for the standard ResNet50 pretrained on ImageNet. <br>
???+ abstract "Reference - Implementation"
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) <br>

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
from torchvision.models import resnet50 as BaseModel
from torchvision.models import ResNet50_Weights
import torch.nn as nn

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#            Architecture class: ResNet50             #
# -----------------------------------------------------#
class ResNet50(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(self, channels, input_resolution=(224, 224), pretrained_weights=False):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    # ---------------------------------------------#
    #         Architecture Attributes             #
    # ---------------------------------------------#
    def get_output_shape(self):
        # Hybrid: use the fast integer division for common input sizes,
        # otherwise run a single non-pretrained forward pass and cache the
        # result to ensure correctness for arbitrary resolutions.
        if hasattr(self, "_cached_output_shape") and self._cached_output_shape:
            return self._cached_output_shape

        common = {(224, 224): (7, 7, 2048)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        import torch

        # Fallback: run a non-pretrained base model to compute actual shape
        full_model = BaseModel(weights=None)
        base_model = nn.Sequential(*(list(full_model.children())[:-2]))
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
        weights = ResNet50_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def create_model(self):
        if self.pretrained_weights:
            weights_arg = ResNet50_Weights.DEFAULT
        else:
            weights_arg = None

        full_model = BaseModel(weights=weights_arg)
        # Return everything up to the final pooling layer
        base_model = nn.Sequential(*(list(full_model.children())[:-2]))
        return base_model
