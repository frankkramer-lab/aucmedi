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
#                    Documentation                     #
# -----------------------------------------------------#
"""The classification variant of the ConvNeXt Base architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ConvNeXtBase"          |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                    |

Recommended alternative `Input_shape` is 384x384 pixels.

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_base.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_base.html) <br>

???+ abstract "Reference - Publication"
    Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
    10 Jan 2022. A ConvNet for the 2020s.
    <br>
    [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)
"""
# -----------------------------------------------------#
#                   Library imports                    #
# -----------------------------------------------------#
# External libraries
from torchvision.models import convnext_base as BaseModel
from torchvision.models import ConvNeXt_Base_Weights
import torchvision.transforms as transforms_module

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#          Architecture class: ConvNeXtBase            #
# -----------------------------------------------------#
class ConvNeXtBase(Architecture_Base):
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
        # ConvNeXt Base has a fixed 32x downsampling ratio
        # Output channels are always 1024 for the base model
        h_out = self.input_shape[0] // 32
        w_out = self.input_shape[1] // 32
        return (h_out, w_out, 1024)

    def get_preprocess(self):
        # https://docs.pytorch.org/vision/stable/models.html
        # Return the weights transforms which include all preprocessing
        weights = ConvNeXt_Base_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                  #
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
