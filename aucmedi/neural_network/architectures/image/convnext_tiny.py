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
"""The classification variant of the ConvNeXt Tiny architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ConvNeXtTiny"          |
| Input_shape              | (224, 224)                 |
| Standardization          | "torch"                   |

Recommended alternative `Input_shape` is 384x384 pixels.

???+ abstract "Reference - Implementation"
    [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) <br>

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
from torch import nn
from torchvision.models import convnext_tiny as TorchvisionModel
from torchvision.models import ConvNeXt_Tiny_Weights

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#          Architecture class: ConvNeXtTiny           #
# -----------------------------------------------------#
class ConvNeXtTiny(Architecture_Base):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(
        self,
        channels=3,
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
        # ConvNeXt Tiny has a fixed 32x downsampling ratio
        # Output channels are 768 for the tiny model
        h_out = self.input_shape[0] // 32
        w_out = self.input_shape[1] // 32
        return (h_out, w_out, 768)

    def get_preprocess(self):
        # Return the weights transforms which include all preprocessing
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        return weights.transforms()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def rechannel_first_layer(self, model):
        # If the number of input channels is not 3, we need to rechannel the first layer
        if self.channels == 3:
            return model
        first_layer = model[0][0]  # Access the first convolutional layer
        # Extract the first conv layer's parameters
        num_filters = model[0][0].out_channels
        kernel_size = model[0][0].kernel_size
        stride = model[0][0].stride
        padding = model[0][0].padding
        # initialize a new convolutional layer
        new_first_layer = nn.Conv2d(
            self.channels,
            num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
        original_weights = model[0][0].weight.data.mean(dim=1, keepdim=True)
        # Expand the averaged weights to the number of input channels of the new dataset
        new_first_layer.weight.data = original_weights.repeat(1, self.channels, 1, 1)
        model[0][0] = new_first_layer
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
