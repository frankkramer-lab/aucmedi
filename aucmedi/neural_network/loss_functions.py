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
import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------#
#                 Focal Loss - Binary                 #
# -----------------------------------------------------#
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Expected input: y_true shape (batch_size, 1) and y_pred shape (batch_size, 1)
        # Verify shapes
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} and y_true shape {y_true.shape} must be the same."
            )

        loss = binary_focal_loss(alpha=self.alpha, gamma=self.gamma)(y_true, y_pred)
        return loss


def binary_focal_loss(alpha=0.25, gamma=2.0):
    """Binary form of focal loss computation.

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
        loss (Loss Function):               A PyTorch compatible loss function. This object can be
                                            passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] `loss` parameter.
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = y_true.float()
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = 1e-7
        # Clip the prediction value
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = torch.ones_like(y_true) * alpha
        alpha_t = torch.where(y_true == 1, alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -torch.log(p_t)
        weight = alpha_t * torch.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = torch.mean(torch.sum(loss, dim=1))
        return loss

    return binary_focal_loss_fixed


# -----------------------------------------------------#
#              Focal Loss - Categorical               #
# -----------------------------------------------------#
class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super(CategoricalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Expected input: y_true shape (batch_size, n_classes) and y_pred shape (batch_size, n_classes)
        loss = categorical_focal_loss(alpha=self.alpha, gamma=self.gamma)(
            y_true, y_pred
        )
        return loss


def categorical_focal_loss(alpha, gamma=2.0):
    """Softmax version of focal loss.

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
        my_loss = CategoricalFocalLoss(alpha=cw_loss)

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
        loss (Loss Function):               A PyTorch compatible loss function. This object can be
                                            passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] `loss` parameter.
    """
    alpha = torch.tensor(alpha, dtype=torch.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_true = y_true.float()
        # Move alpha to the same device as y_pred
        alpha_device = alpha.to(y_pred.device)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate Focal Loss
        loss = alpha_device * torch.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return torch.mean(torch.sum(loss, dim=-1))

    return categorical_focal_loss_fixed


# -----------------------------------------------------#
#               Focal Loss - Multilabel               #
# -----------------------------------------------------#
def multilabel_focal_loss(class_weights, gamma=2.0, class_sparsity_coefficient=1.0):
    """Focal loss for multi-label classification.

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
        loss (Loss Function):               A PyTorch compatible loss function. This object can be
                                            passed to the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] `loss` parameter.
    """
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    class_sparsity_coefficient = torch.tensor(
        class_sparsity_coefficient, dtype=torch.float32
    )

    def focal_loss_function(y_true, y_pred):
        y_true = y_true.float()

        predictions_0 = (1.0 - y_true) * y_pred
        predictions_1 = y_true * y_pred

        cross_entropy_0 = (1.0 - y_true) * (
            -torch.log(torch.clamp(1.0 - predictions_0, 1e-7, 1.0 - 1e-7))
        )
        cross_entropy_1 = y_true * (
            class_sparsity_coefficient
            * -torch.log(torch.clamp(predictions_1, 1e-7, 1.0 - 1e-7))
        )

        cross_entropy = cross_entropy_1 + cross_entropy_0
        class_weighted_cross_entropy = cross_entropy * class_weights

        weight_1 = torch.pow(torch.clamp(1.0 - predictions_1, 1e-7, 1.0 - 1e-7), gamma)
        weight_0 = torch.pow(torch.clamp(predictions_0, 1e-7, 1.0 - 1e-7), gamma)

        weight = weight_0 + weight_1
        focal_loss_tensor = weight * class_weighted_cross_entropy

        return torch.mean(focal_loss_tensor, dim=1)

    return focal_loss_function
