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
# External Libraries
import numpy as np
import torch
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#           Saliency Maps / Backpropagation           #
#-----------------------------------------------------#
class SaliencyMap(XAImethod_Base):
    """ XAI Method for Saliency Map (also called Backpropagation).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Author: Jacob Gil <br>
        GitHub Profile: [https://github.com/jacobgil](https://github.com/jacobgil) <br>
        Date: 2021 <br>
        [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) <br>

    ??? abstract "Reference - Publication"
        Karen Simonyan, Andrea Vedaldi, Andrew Zisserman. 20 Dec 2013.
        Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.
        <br>
        [https://arxiv.org/abs/1312.6034](https://arxiv.org/abs/1312.6034)

    This class provides functionality for running the compute_heatmap function,
    which computes a Saliency Map for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a Saliency Map as XAI Method object.

        Args:
            model (nn.Module):   PyTorch model object.
            layerName (str):                   Not required in Saliency Maps, but defined by Abstract Base Class.
        """
        # Cache class parameters
        self.model = model

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Saliency Map for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Saliency Map for provided image.
        """
        # Convert image to a tensor on the same device as the model
        device = next(self.model.parameters()).device
        inputs = torch.as_tensor(image, dtype=torch.float32, device=device)
        # Track the gradient with respect to the image
        inputs = inputs.detach().clone().requires_grad_(True)

        # Cache & switch training mode for a deterministic forward pass
        was_training = self.model.training
        self.model.eval()

        try:
            # Compute gradient for desierd class index
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = preds[:, class_index].sum()
            loss.backward()
            gradient = inputs.grad
        finally:
            if was_training:
                self.model.train()

        # Obtain maximum gradient of the channel axis
        gradient = gradient.max(dim=1)[0]
        # Convert to NumPy & Remove batch axis
        heatmap = gradient.detach().cpu().numpy()[0]

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
