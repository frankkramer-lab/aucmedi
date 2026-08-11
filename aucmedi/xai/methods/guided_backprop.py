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
#                Guided Backpropagation               #
#-----------------------------------------------------#
class GuidedBackpropagation(XAImethod_Base):
    """ XAI Method for Guided Backpropagation.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: Conor O'Sullivan <br>
        Date: Apr 9, 2025 <br>
        [https://adataodyssey.com/guided-backpropagation/](https://adataodyssey.com/guided-backpropagation/) <br>

    ??? abstract "Reference - Implementation #2"
        Author: Jacob Gil <br>
        GitHub Profile: [https://github.com/jacobgil](https://github.com/jacobgil) <br>
        Date: 2021 <br>
        [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) <br>

    ??? abstract "Reference - Publication"
        Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. 21 Dec 2014.
        Striving for Simplicity: The All Convolutional Net.
        <br>
        [https://arxiv.org/abs/1412.6806](https://arxiv.org/abs/1412.6806)

    This class provides functionality for running the compute_heatmap function,
    which computes a Guided Backpropagation for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating Guided Backpropagation as XAI Method object.

        Args:
            model (nn.Module):                  PyTorch model object.
            layerName (str):                   Not required in Guided Backpropagation, but defined by Abstract Base Class.
        """
        # Cache class parameters
        self.model = model

    #---------------------------------------------#
    #             Guided ReLU Backward            #
    #---------------------------------------------#
    def guided_relu_hook(self, module, grad_input, grad_output):
        """ Internal function. Backward hook which is applied on all ReLU layers.

        Guided Backpropagation only backpropagates positive gradients. As PyTorch already
        masked out the gradients of negative layer inputs, clipping the negative gradients
        results in the guided gradient: `(input > 0) * (gradient > 0) * gradient`.

        Args:
            module (nn.Module):                 The ReLU layer on which the hook is applied.
            grad_input (tuple of torch.Tensor): Gradients with respect to the input of the layer.
            grad_output (tuple of torch.Tensor):Gradients with respect to the output of the layer.

        Returns:
            grad_input (tuple of torch.Tensor): Guided gradients replacing the original ones.
        """
        # Clip negative gradients while preserving unused (None) gradients
        return tuple(None if grad is None else torch.clamp(grad, min=0.0)
                     for grad in grad_input)

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Guided Backpropagation for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        ???+ attention
            Guided Backpropagation is applied on all `torch.nn.ReLU` layers of the model.
            Architectures calling the functional API (`torch.nn.functional.relu`) or
            utilizing other activation functions result in a plain backpropagation.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D or 3D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Guided Backpropagation for provided image.
        """
        # Convert image to a tensor on the same device as the model
        device = next(self.model.parameters()).device
        inputs = torch.as_tensor(image, dtype=torch.float32, device=device)
        # Track the gradient with respect to the image
        inputs = inputs.detach().clone().requires_grad_(True)

        # Identify all ReLU layers on which the guided backpropagation is applied
        relu_layers = [layer for layer in self.model.modules()
                       if isinstance(layer, torch.nn.ReLU)]
        # Cache & deactivate in-place computation, which is unsupported by backward hooks
        inplace_cache = [layer.inplace for layer in relu_layers]
        handles = []
        for layer in relu_layers:
            layer.inplace = False
            handles.append(layer.register_full_backward_hook(self.guided_relu_hook))

        # Cache & switch training mode for a deterministic forward pass
        was_training = self.model.training
        self.model.eval()

        try:
            # Compute gradient for desired class index
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = preds[:, class_index].sum()
            loss.backward()
            gradient = inputs.grad
        finally:
            for handle in handles:
                handle.remove()
            for layer, inplace in zip(relu_layers, inplace_cache):
                layer.inplace = inplace
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
