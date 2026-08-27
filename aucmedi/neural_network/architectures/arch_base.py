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
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from abc import ABC, abstractmethod


# -----------------------------------------------------#
#     Abstract Interface for an Architecture class    #
# -----------------------------------------------------#
class Architecture_Base(ABC):
    """An abstract base class for an Architecture class.

    This class provides functionality for running the create_model function,
    which returns a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

    ???+ example "Create a custom Architecture"
        ```python
        from aucmedi.neural_network.architectures import Architecture_Base
        from torchvision.models import convnext_base as BaseModel
        from torchvision.models import ConvNeXt_Base_Weights

        class My_custom_Architecture(Architecture_Base):
            def __init__(self, channels, input_resolution=(224, 224),
                         pretrained_weights=False):
                self.input = input_resolution + (channels,)
                self.pretrained_weights = pretrained_weights

            def get_output_shape(self):
                # ConvNeXt Base has a fixed 32x downsampling ratio
                # Output channels are always 1024 for the base model
                h_out = self.input_shape[0] // 32
                w_out = self.input_shape[1] // 32
                return (h_out, w_out, 1024)

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
        ```

    ???+ info "Required Functions"
        | Function            | Description                                    |
        | ------------------- | ---------------------------------------------- |
        | `__init__()`        | Object creation function.                      |
        | `create_model()`    | Creating and returning the architecture model. |
    """

    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    @abstractmethod
    def __init__(
        self,
        channels,
        input_resolution=(224, 224),
        pretrained_weights=False,
    ):
        """Functions which will be called during the Architecture object creation.

        This function can be used to pass variables and options in the Architecture instance.

        There are some mandatory required parameters for the initialization: the number of channels, and the
        input resolution (x, y) for an image architecture or (x, y, z) for a volume architecture.

        Args:
            channels (int):                     Number of channels. For example: Grayscale->1 or RGB->3.
            input_resolution (tuple):           Input resolution of the image data for the first model layer (excluding channel axis).
            pretrained_weights (bool):          Option whether to utilize pretrained weights e.g. for ImageNet.
        """
        self.input = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights

    # ---------------------------------------------#
    #         Architecture Attributes              #
    # ---------------------------------------------#
    @abstractmethod
    def get_output_shape(self):
        """Return the output shape of the architecture before the classification head.

        This function will be called inside the AUCMEDI model class to determine the input shape
        for building the classification head.

        Returns:
            output_shape (tuple):            A tuple representing the output shape (height, width, channels)
                                             for image architectures or (depth, height, width, channels)
                                             for volume architectures.
        """
        return None

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    @abstractmethod
    def create_model(self):
        """Create the deep learning or convolutional neural network model.

        This function will be called inside the AUCMEDI model class and have to return a functional
        base model (without the classification head). The model itself should be created here or in a subfunction called
        by this function.

        Returns:
            model (torch.nn.Module):            A PyTorch model.
        """
        return None
