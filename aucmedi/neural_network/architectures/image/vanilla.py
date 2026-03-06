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
| Key in architecture_dict | "2D.Vanilla"               |
| Input_shape              | (224, 224)                 |
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
        channels,
        input_resolution=(224, 224),
        pretrained_weights=False,
    ):
        super(Vanilla, self).__init__()
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

        # Build convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.ReLU()

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def create_model(self):
        return self

    def get_output_shape(self):
        # Calculate output shape after 4x conv + maxpool layers
        h, w, c = self.input_shape
        for _ in range(4):
            h = (h + 1) // 2  # MaxPool with pool_size=2
            w = (w + 1) // 2
        return (h, w, 256)  # 256 filters in the last conv layer

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
