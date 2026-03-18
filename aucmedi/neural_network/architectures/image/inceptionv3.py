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
"""The classification variant of the InceptionV3 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.InceptionV3"           |
| Input_shape              | (299, 299)                 |
| Standardization          | "torch"                    |

???+ abstract "Reference - Implementation"
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html) <br>

???+ abstract "Reference - Publication"
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. 2 Dec 2015.
    Rethinking the Inception Architecture for Computer Vision.
    <br>
    [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
"""
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from torchvision.models import inception_v3 as BaseModel
from torchvision.models import Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#           Architecture class: InceptionV3           #
# -----------------------------------------------------#
class InceptionV3(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(self, channels, input_resolution=(299, 299), pretrained_weights=False):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    # ---------------------------------------------#
    #         Architecture Attributes             #
    # ---------------------------------------------#
    def get_output_shape(self):
        # Hybrid strategy: fast-path known common sizes, otherwise compute
        # by running a single dummy forward pass and cache the result.
        # This avoids repeated expensive model construction/forward calls
        # while remaining correct for arbitrary input sizes.

        # cache per-instance
        if hasattr(self, "_cached_output_shape") and self._cached_output_shape:
            return self._cached_output_shape

        # fast-path for the two most common sizes
        common = {(224, 224): (5, 5, 2048), (299, 299): (8, 8, 2048)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        # fallback: build a non-pretrained model and run a single forward on CPU
        import torch

        full_model = BaseModel(weights=None, aux_logits=False)
        return_nodes = {"Mixed_7c": "features"}
        extractor = create_feature_extractor(full_model, return_nodes=return_nodes)
        extractor = extractor.cpu()
        extractor.eval()
        with torch.no_grad():
            x = torch.zeros(
                1, self.input_shape[2], self.input_shape[0], self.input_shape[1]
            )
            out = extractor(x)

        out_t = out["features"] if isinstance(out, dict) else out

        h_out = int(out_t.shape[2])
        w_out = int(out_t.shape[3])
        c_out = int(out_t.shape[1])
        self._cached_output_shape = (h_out, w_out, c_out)
        return self._cached_output_shape

    def get_preprocess(self):
        weights = Inception_V3_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#

    def create_model(self):
        if self.pretrained_weights:
            model_weights = Inception_V3_Weights.DEFAULT
            weights_arg = model_weights
        else:
            weights_arg = None

        full_model = BaseModel(weights=weights_arg, aux_logits=False)

        # Create a feature extractor that returns the last mixed block
        return_nodes = {"Mixed_7c": "features"}
        feat_extractor = create_feature_extractor(full_model, return_nodes=return_nodes)

        class _FeatureWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor

            def forward(self, x):
                out = self.extractor(x)
                # extractor returns a dict with key 'features'
                return out["features"]

        return _FeatureWrapper(feat_extractor)
