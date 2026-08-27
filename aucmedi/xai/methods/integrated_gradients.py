#==============================================================================#
#  Author:       Dominik Müller                                                #
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
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External Libraries
import numpy as np
import torch
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                 Integrated Gradients                #
#-----------------------------------------------------#
class IntegratedGradients(XAImethod_Base):
    """ XAI Method for Integrated Gradients.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Captum - Model Interpretability for PyTorch <br>
        [https://captum.ai/api/integrated_gradients.html](https://captum.ai/api/integrated_gradients.html) <br>

    ??? abstract "Reference - Publication"
        Mukund Sundararajan, Ankur Taly, Qiqi Yan. 04 Mar 2017.
        Axiomatic Attribution for Deep Networks.
        <br>
        [https://arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365)

    This class provides functionality for running the compute_heatmap function,
    which computes an Integrated Gradients Map for an image with a model.
    """
    def __init__(self, model, layerName=None, num_steps=50):
        """ Initialization function for creating a Integrated Gradients Map as XAI Method object.

        Args:
            model (nn.Module):              PyTorch model object.
            layerName (str):                Not required in Integrated Gradients Maps, but defined by Abstract Base Class.
            num_steps (int):                Number of iterations for interpolation.
        """
        # Cache class parameters
        self.model = model
        self.num_steps = num_steps

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Integrated Gradients Map for a provided image and for specific classification outcome.

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
            heatmap (numpy.ndarray):            Computed Integrated Gradients Map for provided image.
        """
        # Convert image to a tensor on the same device as the model
        device = next(self.model.parameters()).device
        image = torch.as_tensor(image, dtype=torch.float32, device=device).detach()
        # Perform interpolation
        baseline = torch.zeros_like(image)

        # Cache & switch training mode for a deterministic forward pass
        was_training = self.model.training
        self.model.eval()

        # Get the gradients for each interpolated image
        grads = []
        try:
            for step in range(0, self.num_steps + 1):
                cii = baseline + (step / self.num_steps) * (image - baseline)
                # Track the gradient with respect to the interpolated image
                inputs = cii.detach().clone().requires_grad_(True)
                # Compute gradient
                self.model.zero_grad()
                preds = self.model(inputs)
                loss = preds[:, class_index].sum()
                loss.backward()
                # Add to gradient list
                grads.append(inputs.grad[0])
        finally:
            if was_training:
                self.model.train()
        grads = torch.stack(grads)

        # Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = grads.mean(dim=0)
        # Calculate integrated gradients
        integrated_grads = (image - baseline) * avg_grads
        # Obtain maximum gradient of the channel axis
        integrated_grads = integrated_grads.max(dim=1)[0]

        # Convert to NumPy & Remove batch axis
        heatmap = integrated_grads.detach().cpu().numpy()[0]
        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
