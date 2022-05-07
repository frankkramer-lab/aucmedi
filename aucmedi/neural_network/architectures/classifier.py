#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers

#-----------------------------------------------------#
#                 Classification Head                 #
#-----------------------------------------------------#
class Classifier:
    """ A powerful interface for all types of image classifications.

    This class will be created automatically inside the [Neural_Network][aucmedi.neural_network.model.Neural_Network] class.

    !!! info "Supported Features"
        - Binary classification
        - Multi-class classification
        - Multi-label classification
        - 2D/3D data
        - Metadata encoded as NumPy arrays (int or float)

    This class provides functionality for building a classification head for an
    [Architecture][aucmedi.neural_network.architectures]
    ([tensorflow.keras model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)).
    A initialized classifier interface is passed to an architecture class.
    The `build()` function of the classification head is called in the `create_model()`
    function of the architecture.

    !!! info "Structure of the AUCMEDI Classification Head"
        | Layer                         | Description                                                      |
        | ----------------------------- | ---------------------------------------------------------------- |
        | GlobalAveragePooling          | Pooling from Architecture Output to a single spatial dimensions. |
        | Dense(units=512)              | Optional dense & dropout layer if `fcl_dropout=True`.            |
        | Dropout(0.3)                  | Optional dense & dropout layer if `fcl_dropout=True`.            |
        | Dense(units=n_labels)         | Dense layer to the number of labels (classes).                   |
        | Activation(activation_output) | Activation function corresponding to classification type.        |

    ???+ note "Classification Types"
        | Type                       | Activation Function                                             |
        | -------------------------- | --------------------------------------------------------------- |
        | Binary classification      | `activation_output="softmax"`: Only a single class is correct.  |
        | Multi-class classification | `activation_output="softmax"`: Only a single class is correct.  |
        | Multi-label classification | `activation_output="sigmoid"`: Multiple classes can be correct. |

        For more information on multi-class vs multi-label, check out this blog post from Rachel Draelos: <br>
        https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/

    The recommended way is to pass all required variables to the [Neural_Network][aucmedi.neural_network.model.Neural_Network]
    which automatically creates the Classifier and passes it to the Architecture.

    ???+ example
        ```python
        # Recommended way (automatic creation in Neural_Network)
        model = Neural_Network(n_labels=20, channels=3, batch_queue_size=1,
                               input_shape=(32, 32), activation_output="sigmoid",
                               fcl_dropout=False)

        # Manual way
        from aucmedi.neural_network.architectures import Classifier
        from aucmedi.neural_network.architectures.image import Architecture_Vanilla

        classification_head = Classifier(n_labels=20, fcl_dropout=True,
                                         activation_output="sigmoid")
        arch = Architecture_Vanilla(classification_head, channels=3,
                                    input_shape=(32, 32))
        ```
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_labels, fcl_dropout=True, activation_output="softmax"):
        """ Initialization function for creating a Classifier object.

        The fully connected layer and dropout option (`fcl_dropout`) utilizes a 512 unit Dense layer with 30% Dropout.

        Modi for activation_output: Check out [TensorFlow.Keras doc on activation functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations).

        Args:
            n_labels (int):                 Number of classes/labels (important for the last layer of classification head).
            fcl_dropout (bool):             Option whether to utilize a Dense & Dropout layer before the last classification layer.
            activation_output (str):        Activation function which is used in the last classification layer.
        """
        self.n_labels = n_labels
        self.fcl_dropout = fcl_dropout
        self.activation_output = activation_output

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def build(self, model_input, model_output, two_dim=True):
        """ Internal function which appends the classification head.

        This function will be called inside of an [Architecture][aucmedi.neural_network.architectures] `create_model()` function
        and must return a functional Keras model.
        The `build()` function will append a classification head to the provided Keras model.

        Args:
            model_input (tf.keras layer):       Input layer of the model.
            model_output (tf.keras layer):      Output layer of the model.
            two_dim (bool):                     Boolean, whether architecture is 2D or 3D.

        Returns:
            model (tf.keras model):             A functional Keras model.
        """
        # Apply GlobalAveragePooling to obtain a single spatial dimensions
        if two_dim:
            model_head = layers.GlobalAveragePooling2D(name="avg_pool")(model_output)
        else:
            model_head = layers.GlobalAveragePooling3D(name="avg_pool")(model_output)

        # Apply optional dense & dropout layer
        if self.fcl_dropout:
            model_head = layers.Dense(units=512)(model_head)
            model_head = layers.Dropout(0.3)(model_head)

        # Apply classifier
        model_head = layers.Dense(self.n_labels, name="preds")(model_head)
        # Apply activation output according to classification type
        model_head = layers.Activation(self.activation_output, name="probs")(model_head)

        # Create tf.keras model
        model = Model(inputs=model_input, outputs=model_head)

        # Return ready-to-use classifier model
        return model
