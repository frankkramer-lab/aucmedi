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
from abc import ABC, abstractmethod
import numpy as np
import torch

#-----------------------------------------------------#
#         Abstract Base Class for XAI Methods         #
#-----------------------------------------------------#
class XAImethod_Base(ABC):
    """ An abstract base class for a XAI Method class.

    Normally, this class is used internally in the xai_decoder function in the AUCMEDI XAI module.

    This class provides functionality for running the compute_heatmap function,
    which computes a heatmap for an image with a model.

    ???+ example "Create a custom XAImethod"
        ```python
        from aucmedi.xai.methods.xai_base import XAImethod_Base

        class My_custom_XAImethod(XAImethod_Base):
            def __init__(self, model, layerName=None):
                pass

            def compute_heatmap(self, image, class_index, eps=1e-8):
                pass
        ```

    ???+ info "Required Functions"
        | Function            | Description                                |
        | ------------------- | ------------------------------------------ |
        | `__init__()`        | Object creation function.                  |
        | `compute_heatmap()` | Application of the XAI Method on an image. |

    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    @abstractmethod
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a XAI Method object.
        ```
        __init__(model, layerName=None)
        ```

        Args:
            model (nn.Module):   PyTorch model object.
            layerName (str):                Layer name of the convolutional layer for heatmap computation.
        """
        pass

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the XAI heatmap for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap should be encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed XAI heatmap for provided image.
        """
        pass

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
def to_numpy(image):
    """ Internal function. Converts an image batch into a NumPy matrix.

    Args:
        image (numpy.ndarray or torch.Tensor):  Image matrix provided as one-element batch.

    Returns:
        image (numpy.ndarray):                  Image matrix encoded as NumPy Array.
    """
    if torch.is_tensor(image):
        return image.detach().cpu().numpy()
    return np.asarray(image)

def predict_proba(model, batch):
    """ Internal function. Computes the class probabilities for a batch of images.

    Perturbation based XAI methods (e.g. Occlusion Sensitivity or LIME) require class
    confidences instead of gradients. As the AUCMEDI architectures return raw logits,
    a softmax normalization is applied, which corresponds to the default
    `activation_output` of the [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork].

    Args:
        model (nn.Module):              PyTorch model object.
        batch (numpy.ndarray):          Batch of images encoded channel-first: (N,C,H,W) / (N,C,D,H,W).

    Returns:
        preds (numpy.ndarray):          Class probabilities with shape (N, n_labels).
    """
    # Convert batch to a tensor on the same device as the model
    device = next(model.parameters()).device
    inputs = torch.as_tensor(np.asarray(batch), dtype=torch.float32, device=device)
    # Cache & switch training mode for a deterministic forward pass
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            preds = torch.softmax(model(inputs), dim=1)
    finally:
        if was_training:
            model.train()
    # Return the class probabilities
    return preds.cpu().numpy()
