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
"""The classification variant of the DenseNet201 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.DenseNet201"           |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html) <br>

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
from torchvision.models import densenet201 as BaseModel
from torchvision.models import DenseNet201_Weights
import torchvision.transforms as transforms_module

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#           Architecture class: DenseNet201           #
# -----------------------------------------------------#
class DenseNet201(Architecture_Base):
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
        # DenseNet reduces spatial resolution by a factor of 32
        # Output channels are 1920 for DenseNet201
        # Hybrid: fast-path for common size, otherwise compute via one-time
        # non-pretrained forward pass and cache the result.
        if hasattr(self, "_cached_output_shape") and self._cached_output_shape:
            return self._cached_output_shape

        common = {(224, 224): (7, 7, 1920)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        import torch

        full_model = BaseModel(weights=None)
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
        weights = DenseNet201_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#

    def create_model(self):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights:
            model_weights = "DEFAULT"
        else:
            model_weights = None

        # Obtain base model (omit classification head)
        full_model = BaseModel(weights=model_weights)
        base_model = full_model.features
        return base_model
