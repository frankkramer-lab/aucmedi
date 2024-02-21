#==============================================================================#
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
#==============================================================================#
#-----------------------------------------------------#
#                    Documentation                    #
#-----------------------------------------------------#
""" Models are represented with the open-source neural network library [TensorFlow.Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
    which provides an user-friendly API for commonly used neural-network building blocks.

The already implemented architectures are configurable by custom input sizes, optional dropouts, transfer learning via pretrained weights,
meta data inclusion or activation output depending on classification type.

Additionally, AUCMEDI offers architectures for 2D image and 3D volume classification.

???+ example "Example: How to select an Architecture"
    For architecture selection, just create a key (str) by adding "2D." or "3D." to the architecture name,
    and pass the key to the `architecture` parameter of the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] class.

    ```python
    # 2D architecture
    my_model_a = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")
    # 3D architecture for multi-label classification (sigmoid activation)
    my_model_b = NeuralNetwork(n_labels=8, channels=3, architecture="3D.ResNet50",
                                activation_output="sigmoid")
    # 2D architecture with custom input_shape
    my_model_c = NeuralNetwork(n_labels=8, channels=3, architecture="2D.Xception",
                                input_shape=(512,512))
    ```

???+ note "List of implemented Architectures"
    AUCMEDI provides a large library of state-of-the-art and ready-to-use architectures.

    - 2D Architectures: [aucmedi.neural_network.architectures.image][]
    - 3D Architectures: [aucmedi.neural_network.architectures.volume][]

Besides the flexibility in switching between already implemented architecture,
the [abstract base class interface][aucmedi.neural_network.architectures.arch_base.Architecture_Base]
for architectures offers the possibility for custom architecture integration into the AUCMEDI pipeline.

Furthermore, AUCMEDI offers the powerful classification head interface [Classifier][aucmedi.neural_network.architectures.classifier],
which can be used for all types of image classifications and will be automatically created in the
[NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] class.
"""
#-----------------------------------------------------#
#               General library imports               #
#-----------------------------------------------------#
# Abstract Base Class for Architectures
from aucmedi.neural_network.architectures.arch_base import Architecture_Base
# Classification Head
from aucmedi.neural_network.architectures.classifier import Classifier

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#
# Initialize combined architecture_dict for image & volume architectures
architecture_dict = {}

# Add image architectures to architecture_dict
from aucmedi.neural_network.architectures.image import architecture_dict as arch_image
for arch in arch_image:
    architecture_dict["2D." + arch] = arch_image[arch]

# Add volume architectures to architecture_dict
from aucmedi.neural_network.architectures.volume import architecture_dict as arch_volume
for arch in arch_volume:
    architecture_dict["3D." + arch] = arch_volume[arch]

#-----------------------------------------------------#
#       Meta Information of Architecture Classes      #
#-----------------------------------------------------#
# Initialize combined supported_standardize_mode for image & volume architectures
supported_standardize_mode = {}

# Add image architectures to supported_standardize_mode
from aucmedi.neural_network.architectures.image import supported_standardize_mode as modes_image
for m in modes_image:
    supported_standardize_mode["2D." + m] = modes_image[m]

# Add volume architectures to supported_standardize_mode
from aucmedi.neural_network.architectures.volume import supported_standardize_mode as modes_volume
for m in modes_volume:
    supported_standardize_mode["3D." + m] = modes_volume[m]
