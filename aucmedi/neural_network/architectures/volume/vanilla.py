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
"""The classification variant of the Vanilla architecture.

No intensive hardware requirements, which makes it ideal for debugging.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "3D.Vanilla"               |
| Input_shape              | (128, 128, 128)            |
| Standardization          | "z-score"                  |

???+ abstract "Reference - Implementation"
    [https://github.com/wanghsinwei/isic-2019/](https://github.com/wanghsinwei/isic-2019/) <br>
"""
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from torch import nn

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#                 Vanilla Architecture                #
# -----------------------------------------------------#
class Vanilla(nn.Module, Architecture_Base):
    # ---------------------------------------------#
    #                   __init__                  #
    # ---------------------------------------------#
    def __init__(
        self,
        classification_head,
        channels,
        input_shape=(128, 128, 128),
        pretrained_weights=False,
    ):
        super(Vanilla, self).__init__()
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

        # Build convolutional layers
        self.conv1 = nn.Conv3d(channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.relu4 = nn.ReLU()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def create_model(self):
        return self

    def output_shape(self):
        # Calculate output shape after 4x conv + maxpool layers
        d, h, w, c = self.input
        for _ in range(4):
            d = (d + 1) // 2  # MaxPool with pool_size=2
            h = (h + 1) // 2
            w = (w + 1) // 2
        return (d, h, w, 256)  # 256 filters in the last conv layer

    def forward(self, x):
        # Conv Block 1
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        # Conv Block 2
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        # Conv Block 3
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)

        # Conv Block 4
        x = self.relu4(self.conv4(x))
        x = self.pool4(x)

        return x


"""
    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def create_model(self):
        # Initialize input
        model_input = Input(shape=self.input)

        # Add 4x convolutional layers with increasing filters
        model_base = Conv3D(
            filters=32, kernel_size=3, padding="same", activation="relu"
        )(model_input)
        model_base = MaxPooling3D(pool_size=2)(model_base)

        model_base = Conv3D(
            filters=64, kernel_size=3, padding="same", activation="relu"
        )(model_base)
        model_base = MaxPooling3D(pool_size=2)(model_base)

        model_base = Conv3D(
            filters=128, kernel_size=3, padding="same", activation="relu"
        )(model_base)
        model_base = MaxPooling3D(pool_size=2)(model_base)

        model_base = Conv3D(
            filters=256, kernel_size=3, padding="same", activation="relu"
        )(model_base)
        model_base = MaxPooling3D(pool_size=2)(model_base)

        # Add classification head
        model = self.classifier.build(model_input=model_input, model_output=model_base)

        # Return created model
        return model
"""
