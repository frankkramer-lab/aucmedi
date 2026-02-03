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
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import torch
import torch.nn as nn


# -----------------------------------------------------#
#                 Classification Head                 #
# -----------------------------------------------------#
class Classifier(nn.Module):
    """A powerful interface for all types of image classifications.

    This class will be created automatically inside the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] class.

    !!! info "Supported Features"
        - Binary classification
        - Multi-class classification
        - Multi-label classification
        - 2D/3D data
        - Metadata encoded as NumPy arrays (int or float)

    This class provides functionality for building a classification head for an
    [Architecture][aucmedi.neural_network.architectures]
    ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html)).
    A initialized classifier interface is passed to an architecture class.
    The `build()` function of the classification head is called in the `create_model()`
    function of the architecture.

    !!! info "Structure of the AUCMEDI Classification Head"
        | Layer                         | Description                                                      |
        | ----------------------------- | ---------------------------------------------------------------- |
        | AdaptiveAvgPool               | Pooling from Architecture Output to a single spatial dimensions. |
        | Linear(512)                   | Optional linear & dropout layer if `fcl_dropout=True`.            |
        | Dropout(0.3)                  | Optional linear & dropout layer if `fcl_dropout=True`.            |
        | Concatenate()                 | Optional appending of metadata to classification head.           |
        | Linear(512)                   | Optional linear & dropout layer if metadata is present.           |
        | Dropout(0.3)                  | Optional linear & dropout layer if metadata is present.           |
        | Linear(256)                   | Optional linear & dropout layer if metadata is present.           |
        | Dropout(0.3)                  | Optional linear & dropout layer if metadata is present.           |
        | Linear(n_labels)              | Linear layer to the number of labels (classes).                   |
        | Activation(activation_output) | Activation function corresponding to classification type.        |

    ???+ note "Classification Types"
        | Type                       | Activation Function                                             |
        | -------------------------- | --------------------------------------------------------------- |
        | Binary classification      | `activation_output="softmax"`: Only a single class is correct.  |
        | Multi-class classification | `activation_output="softmax"`: Only a single class is correct.  |
        | Multi-label classification | `activation_output="sigmoid"`: Multiple classes can be correct. |

        For more information on multi-class vs multi-label, check out this blog post from Rachel Draelos: <br>
        [https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/](https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/)

    The recommended way is to pass all required variables to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork]
    which automatically creates the Classifier and passes it to the Architecture.

    ???+ example
        ```python
        # Recommended way (automatic creation in NeuralNetwork)
        model = NeuralNetwork(n_labels=20, channels=3,
                              input_shape=(32, 32), activation_output="sigmoid",
                              fcl_dropout=False)

        # Manual way
        from aucmedi.neural_network.architectures import Classifier
        from aucmedi.neural_network.architectures.image import Vanilla

        classification_head = Classifier(n_labels=20, fcl_dropout=False,
                                         activation_output="sigmoid")
        arch = Vanilla(classification_head, channels=3,
                       input_shape=(32, 32))
        ```

    ??? example "Example: How to integrate metadata in AUCMEDI?"
        ```python
        from aucmedi import *
        import numpy as np

        my_metadata = np.random.rand(len(samples), 10)

        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121",
                                  n_meta_variables=10)

        my_dg = DataGenerator(samples, "images_dir/",
                              labels=None, metadata=my_metadata,
                              resize=my_model.meta_input,                  # (224,224)
                              standardize_mode=my_model.arch_standardize)  # "torch"
        ```
    """

    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(
        self,
        n_labels,
        activation_output="softmax",
        n_meta_variables=None,
        fcl_dropout=True,
    ):
        """Initialization function for creating a Classifier object.

        The fully connected layer and dropout option (`fcl_dropout`) utilizes a 512 unit Linear layer with 30% Dropout.

        Modi for activation_output: Check out [PyTorch doc on activation functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).

        Args:
            n_labels (int):                 Number of classes/labels (important for the last layer of classification head).
            activation_output (str):        Activation function which is used in the last classification layer.
            n_meta_variables (int):           Number of metadata variables, which should be included in the classification head.
                                            If `None`is provided, no metadata integration block will be added to the classification head.
            fcl_dropout (bool):             Option whether to utilize a Linear & Dropout layer before the last classification layer.
        """
        super(Classifier, self).__init__()
        self.n_labels = n_labels
        self.activation_output = activation_output
        self.n_meta_variables = n_meta_variables
        self.fcl_dropout = fcl_dropout

        # Define activation
        if self.activation_output is None or self.activation_output == "None":
            self.activation = nn.Identity()
        elif self.activation_output == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif self.activation_output == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation_output")

    # ---------------------------------------------#
    #                Create Model                 #
    # ---------------------------------------------#
    def build(self, model_base, arch_output_shape):
        """Internal function which appends the classification head.

        This function will be called from inside an [Architecture][aucmedi.neural_network.architectures] `create_model()` function
        and must return a functional PyTorch model.
        The `build()` function will append a classification head to the provided PyTorch model.

        Args:
            model_base (torch.nn.Module):    Base model/feature extractor that has an output_shape() method.

        Returns:
            model (torch.nn.Module):         A PyTorch module.
        """
        # Convert to channel dimension to get number of feature maps
        num_channels = arch_output_shape[-1]

        pool_layers = []

        # Apply AdaptiveAvgPool to obtain single spatial dimension
        # This is flexible and works for any number of spatial dimensions
        if (
            len(arch_output_shape) == 3
        ):  # for 2D architectures (height, width, channels)
            pool_layers.append(nn.AdaptiveAvgPool2d(1))
            pool_layers.append(nn.Flatten())
        elif (
            len(arch_output_shape) == 4
        ):  # for 3D architectures (depth, height, width, channels)
            pool_layers.append(nn.AdaptiveAvgPool3d(1))
            pool_layers.append(nn.Flatten())
        else:
            pool_layers.append(nn.Flatten())

        # After adaptive pooling, the feature dimension is just the number of channels
        pooled_feature_dim = num_channels

        # Apply optional linear & dropout layer
        if self.fcl_dropout:
            pool_layers.append(nn.Linear(pooled_feature_dim, 512))
            pool_layers.append(nn.Dropout(0.3))
            current_dim = 512
        else:
            current_dim = pooled_feature_dim
        pool = nn.Sequential(*pool_layers)

        # Apply metadata integration block
        meta_layers = []
        meta = None
        if self.n_meta_variables is not None:
            meta_layers.append(nn.Linear(current_dim + self.n_meta_variables, 512))
            meta_layers.append(nn.ReLU())
            meta_layers.append(nn.Dropout(0.3))
            meta_layers.append(nn.Linear(512, 256))
            meta_layers.append(nn.ReLU())
            meta_layers.append(nn.Dropout(0.3))
            current_dim = 256
            meta = nn.Sequential(*meta_layers)

        # Apply classifier
        class_layers = []
        class_layers.append(nn.Linear(current_dim, self.n_labels))
        class_layers.append(self.activation)

        # Create nn.Sequential for the head
        class_head = nn.Sequential(*class_layers)

        class Classifier_Model(nn.Module):
            def __init__(self, base, pool, head, meta=None):
                super().__init__()
                self.base = base
                self.pool = pool
                self.meta = meta
                self.head = head

            def forward(self, x, metadata=None):
                x = self.base(x)
                # Head handles pooling and flattening
                x = self.pool(x)
                if metadata is not None:
                    x = torch.cat((x, metadata), dim=1)
                    x = self.meta(x)
                x = self.head(x)
                return x

        model = Classifier_Model(model_base, pool, class_head, meta)

        # Return ready-to-use classifier model
        return model
