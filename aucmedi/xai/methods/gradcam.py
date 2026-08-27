# ==============================================================================#
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External Libraries
import numpy as np
import torch

# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base


# -----------------------------------------------------#
#     Gradient-weighted Class Activation Mapping      #
# -----------------------------------------------------#
class GradCAM(XAImethod_Base):
    """XAI Method for Gradient-weighted Class Activation Mapping (Grad-CAM).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: Jacob Gil <br>
        GitHub Profile: [https://github.com/jacobgil](https://github.com/jacobgil) <br>
        Date: 2021 <br>
        [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) <br>

    ??? abstract "Reference - Publication"
        Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. 7 Oct 2016.
        Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
        <br>
        [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

    This class provides functionality for running the compute_heatmap function,
    which computes a Grad-CAM heatmap for an image with a model.
    """

    def __init__(self, model, layerName=None):
        """Initialization function for creating a Grad-CAM as XAI Method object.

        Args:
            model (nn.Module):              PyTorch model object.
            layerName (str):                   Layer name of the convolutional layer for heatmap computation.
        """
        # Cache class parameters
        self.model = model
        self.layerName = layerName
        # If not defined, the output layer is identified on the first
        # compute_heatmap() call, which requires an image for the dry run.

    # ---------------------------------------------#
    #            Identify Output Layer            #
    # ---------------------------------------------#
    def get_by_name(self, model, layerName):
        """Internal function. Resolves the layer name.

        Args:
            model (nn.Module):              PyTorch model object.
            layerName (str):                Layer name in dot notation as by calling model.named_modules().
        Returns:
            layer (nn.Module):              PyTorch layer object.
        """
        # Access by dot-notated name
        parts = layerName.split('.')
        current_module = model
        try:
            for part in parts:
                current_module = getattr(current_module, part)
            return current_module
        except AttributeError:
            return None
    
    def iterate_layers(module):
        for name, layer in module.named_children():
            print(f"Layer Name: {name}, Layer Type: {type(layer)}")
            iterate_layers(layer)  # Recursively iterate over nested layers

    def find_output_layer(self, image):
        """Internal function. Applied if `layerName==None`.

        Identify last/final layer with a feature map output in neural network architecture.
        This layer is used to obtain activation outputs / feature map.

        In contrast to Keras, output shapes of a PyTorch model are only known at runtime,
        which is why the layers are identified via a dry run on the provided image.

        Args:
            image (torch.Tensor):           Image batch used for the dry run.

        Returns:
            layerName (str):                Layer name in dot notation as by calling model.named_modules().
        """
        # Cache all layers with a feature map output during a forward pass
        candidates = []
        handles = []

        def shape_hook(name):
            def hook(module, input, output):
                # Check to see if the layer has a 4D output (batch, channel, spatial axes).
                # Layers collapsing the spatial axes (e.g. the adaptive pooling of the
                # classification head) are skipped, as a 1x1 feature map holds no localization.
                if isinstance(output, torch.Tensor) and output.dim() >= 4 \
                        and min(output.shape[2:]) > 1:
                    candidates.append(name)
            return hook

        # Register hook on all leaf layers (containers only pass through their child output)
        for name, layer in self.model.named_modules():
            if len(list(layer.children())) == 0:
                handles.append(layer.register_forward_hook(shape_hook(name)))
        # Run inference to record the layers in execution order
        try:
            with torch.no_grad():
                self.model(image)
        finally:
            for handle in handles:
                handle.remove()
        # Otherwise, throw exception
        if len(candidates) == 0:
            raise ValueError("Could not find 4D layer. Cannot apply Grad-CAM.")
        # Return last layer with a feature map output
        return candidates[-1]
    
    # ---------------------------------------------#
    #             Feature Map Gradient            #
    # ---------------------------------------------#
    def compute_feature_gradient(self, image, class_index):
        """Internal function. Computes the feature map of the target layer and its gradient.

        Shared by all Grad-CAM based XAI methods, which only differ in how the gradient
        is weighted into a heatmap.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the gradient should be computed.

        Returns:
            conv_out (torch.Tensor):            Feature map output of the target layer.
            grads (torch.Tensor):               Gradient of the class score regarding the feature map.
        """
        # Convert image to a tensor on the same device as the model
        device = next(self.model.parameters()).device
        inputs = torch.as_tensor(image, dtype=torch.float32, device=device)

        # Try to find output layer if not defined
        if self.layerName is None:
            self.layerName = self.find_output_layer(inputs)
        # Resolve target layer for heatmap computation
        target_layer = self.get_by_name(self.model, self.layerName)
        if target_layer is None:
            raise ValueError(f"Layer '{self.layerName}' could not be found "
                              "in the provided model. Cannot apply Grad-CAM.")

        # Register hooks to capture the layer activation & its gradient
        activation = {}
        gradient = {}

        def forward_hook(module, input, output):
            activation["value"] = output

        def backward_hook(module, grad_input, grad_output):
            gradient["value"] = grad_output[0]

        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        # Cache & switch training mode for a deterministic forward pass
        was_training = self.model.training
        self.model.eval()

        try:
            # Compute gradient for desired class index
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = preds[:, class_index].sum()
            loss.backward()

            conv_out = activation["value"]
            grads = gradient["value"]
        finally:
            handle_forward.remove()
            handle_backward.remove()
            if was_training:
                self.model.train()

        # Return the feature map and its gradient
        return conv_out, grads

    # ---------------------------------------------#
    #             Heatmap Computation             #
    # ---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """Core function for computing the Grad-CAM heatmap for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D or 3D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Grad-CAM for provided image.
        """
        # Obtain feature map of the last conv layer and its gradient
        conv_out, grads = self.compute_feature_gradient(image, class_index)

        # Identify pooling axis (all spatial axes, keep batch & channel axis)
        pooling_axis = tuple(range(2, grads.dim()))
        # Averaged output gradient based on feature map of last conv layer
        pooled_grads = grads.mean(dim=pooling_axis)[0]
        # Normalize gradients via "importance"
        conv_out = conv_out[0]
        weights = pooled_grads.view(-1, *([1] * (conv_out.dim() - 1)))
        heatmap = (conv_out * weights).sum(dim=0)
        heatmap = heatmap.detach().cpu().numpy()

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
