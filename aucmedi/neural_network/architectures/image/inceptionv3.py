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
import torch
from torchvision.models import inception_v3 as TorchvisionModel
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
        # The InceptionV3 architecture is designed for 299x299 inputs, so we will upsample smaller inputs in the create_model method.
        # throw warning if input resolution is smaller than 299x299, as this may lead to increased memory usage and slower training
        if input_resolution[0] < 299 or input_resolution[1] < 299:
            print(
                f"Warning: InceptionV3 is designed for input resolution of at least 299x299. Your input resolution is {input_resolution}. The model will upsample inputs to 299x299, which may lead to increased memory usage and slower training."
            )

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

        # fast-path for the most common sizes
        common = {(299, 299): (8, 8, 2048)}
        res = (self.input_shape[0], self.input_shape[1])
        if res in common:
            self._cached_output_shape = common[res]
            return self._cached_output_shape

        # fallback: build a non-pretrained model and run a single forward on CPU
        full_model = TorchvisionModel(weights=None, aux_logits=False)
        return_nodes = {"Mixed_7c": "features"}
        extractor = create_feature_extractor(full_model, return_nodes=return_nodes)
        if self.channels != 3:
            extractor = self.rechannel_first_layer(extractor)
        extractor = extractor.cpu()
        extractor.eval()
        with torch.no_grad():
            h = max(self.input_shape[0], 299)
            w = max(self.input_shape[1], 299)
            x = torch.zeros(1, self.input_shape[2], h, w)
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
    def rechannel_first_layer(self, model):
        # InceptionV3's first conv layer is named Conv2d_1a_3x3.conv
        conv1 = model.Conv2d_1a_3x3.conv
        new_conv1 = nn.Conv2d(
            self.channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )
        with torch.no_grad():
            if self.channels == 1:
                new_conv1.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
            else:
                new_conv1.weight.copy_(
                    conv1.weight.repeat(1, self.channels // 3 + 1, 1, 1)[
                        :, : self.channels
                    ]
                )
            if conv1.bias is not None:
                new_conv1.bias.copy_(conv1.bias)
        model.Conv2d_1a_3x3.conv = new_conv1
        return model

    def create_model(self):
        if self.pretrained_weights:
            model_weights = Inception_V3_Weights.DEFAULT
            weights_arg = model_weights
        else:
            weights_arg = None

        full_model = TorchvisionModel(weights=weights_arg, aux_logits=False)

        # Create a feature extractor that returns the last mixed block
        return_nodes = {"Mixed_7c": "features"}
        feat_extractor = create_feature_extractor(full_model, return_nodes=return_nodes)
        if self.channels != 3:
            feat_extractor = self.rechannel_first_layer(feat_extractor)

        class _FeatureWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor

            def forward(self, x):
                out = self.extractor(x)
                return out["features"]

        base_model = _FeatureWrapper(feat_extractor)

        # If the user provides smaller inputs than the model was designed for,
        # upsample to the recommended minimum to avoid kernel > input errors.
        min_size = 299
        if self.input_shape[0] < min_size or self.input_shape[1] < min_size:
            up = nn.Upsample(
                size=(min_size, min_size), mode="bilinear", align_corners=False
            )
            base_model = nn.Sequential(up, base_model)

        return base_model
