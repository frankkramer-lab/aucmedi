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
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

#-----------------------------------------------------#
#                 Focal Loss - Binary                 #
#-----------------------------------------------------#
def binary_focal_loss(alpha=0.25, gamma=2.0):
    """ Binary form of focal loss computation.

    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    ??? example
        ```python
        from aucmedi.neural_network.loss_functions import *
        my_loss = binary_focal_loss(alpha=0.75)

        model = NeuralNetwork(n_labels=1, channels=3, loss=my_loss)
        ```

    ??? abstract "Reference - Implementation"
        Author: Umberto Griffo <br>
        GitHub: [https://github.com/umbertogriffo](https://github.com/umbertogriffo) <br>
        Source: [https://github.com/umbertogriffo/focal-loss-keras](https://github.com/umbertogriffo/focal-loss-keras) <br>

    ??? abstract "Reference - Publication"
        Focal Loss for Dense Object Detection (Aug 2017) <br>
        Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár <br>
        [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

    Args:
        alpha (float):      Class weight for positive class.
        gamma (float):      Tunable focusing parameter (γ ≥ 0).

    Returns:
        loss (Loss Function):               A TensorFlow compatible loss function. This object can be
                                            passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] `loss` parameter.
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

#-----------------------------------------------------#
#              Focal Loss - Categorical               #
#-----------------------------------------------------#
def categorical_focal_loss(alpha, gamma=2.0):
    """ Softmax version of focal loss.

    When there is a skew between different categories/labels in your data set,
    you can try to apply this function as a loss.

    ```
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation
    ```

    The `class_weights_list` obtained from [compute_class_weights][aucmedi.utils.class_weights.compute_class_weights]
    can be provided as parameter `alpha`.

    ??? example
        ```python
        # Compute class weights
        from aucmedi.utils.class_weights import compute_class_weights
        cw_loss, cw_fit = compute_class_weights(class_ohe)

        from aucmedi.neural_network.loss_functions import *
        my_loss = categorical_focal_loss(alpha=cw_loss)

        model = NeuralNetwork(n_labels=6, channels=3, loss=my_loss)
        ```

    ??? abstract "Reference - Implementation"
        Author: Umberto Griffo <br>
        GitHub: [https://github.com/umbertogriffo](https://github.com/umbertogriffo) <br>
        Source: [https://github.com/umbertogriffo/focal-loss-keras](https://github.com/umbertogriffo/focal-loss-keras) <br>

    ??? abstract "Reference - Publication"
        Focal Loss for Dense Object Detection (Aug 2017) <br>
        Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár <br>
        [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

    Args:
        alpha (list of float):      The same as weighing factor in balanced cross entropy.
                                    Alpha is used to specify the weight of different categories/labels,
                                    the size of the array needs to be consistent with the number of classes.
        gamma (float):              Focusing parameter for modulating factor (1-p).

    Returns:
        loss (Loss Function):               A TensorFlow compatible loss function. This object can be
                                            passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] `loss` parameter.
    """
    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

#-----------------------------------------------------#
#               Focal Loss - Multilabel               #
#-----------------------------------------------------#
def multilabel_focal_loss(class_weights, gamma=2.0,
                          class_sparsity_coefficient=1.0):
    """ Focal loss for multi-label classification.

    ??? example
        ```python
        # Compute class weights
        from aucmedi.utils.class_weights import compute_class_weights
        class_weights = compute_multilabel_weights(class_ohe)

        from aucmedi.neural_network.loss_functions import *
        my_loss = multilabel_focal_loss(class_weights=class_weights)

        model = NeuralNetwork(n_labels=6, channels=3, loss=my_loss,
                               activation_output="sigmoid")
        ```

    ??? abstract "Reference - Implementation"
        Author: Sushant Tripathy <br>
        LinkedIn: [https://www.linkedin.com/in/sushanttripathy/](https://www.linkedin.com/in/sushanttripathy/) <br>
        Source: [https://github.com/sushanttripathy/Keras_loss_functions/blob/master/focal_loss.py](https://github.com/sushanttripathy/Keras_loss_functions/blob/master/focal_loss.py) <br>

    ??? abstract "Reference - Publication"
        Focal Loss for Dense Object Detection (Aug 2017) <br>
        Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár  <br>
        [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002) <br>

    Args:
        class_weights (list of float):      Non-zero, positive class-weights. This is used instead
                                            of alpha parameter.
        gamma (float):                      The Gamma parameter in Focal Loss. Default value (2.0).
        class_sparsity_coefficient (float): The weight of True labels over False labels. Useful
                                            if True labels are sparse. Default value (1.0).
    Returns:
        loss (Loss Function):               A TensorFlow compatible loss function. This object can be
                                            passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] `loss` parameter.
    """
    class_weights = K.constant(class_weights, tf.float32)
    gamma = K.constant(gamma, tf.float32)
    class_sparsity_coefficient = K.constant(class_sparsity_coefficient,
                                            tf.float32)

    def focal_loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        predictions_0 = (1.0 - y_true) * y_pred
        predictions_1 = y_true * y_pred

        cross_entropy_0 = (1.0 - y_true) * (-K.log(K.clip(1.0 - predictions_0,
                                K.epsilon(), 1.0 - K.epsilon())))
        cross_entropy_1 = y_true *(class_sparsity_coefficient * -K.log(K.clip(
                                predictions_1, K.epsilon(), 1.0 - K.epsilon())))

        cross_entropy = cross_entropy_1 + cross_entropy_0
        class_weighted_cross_entropy = cross_entropy * class_weights

        weight_1 = K.pow(K.clip(1.0 - predictions_1,
                                K.epsilon(), 1.0 - K.epsilon()), gamma)
        weight_0 = K.pow(K.clip(predictions_0, K.epsilon(),
                                1.0 - K.epsilon()), gamma)

        weight = weight_0 + weight_1
        focal_loss_tensor = weight * class_weighted_cross_entropy

        return K.mean(focal_loss_tensor, axis=1)

    return focal_loss_function
