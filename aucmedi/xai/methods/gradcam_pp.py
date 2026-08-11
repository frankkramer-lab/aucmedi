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
from aucmedi.xai.methods.gradcam import GradCAM

#-----------------------------------------------------#
#                XAI Method: Grad-Cam++               #
#-----------------------------------------------------#
class GradCAMpp(GradCAM):
    """ XAI Method for Grad-CAM++.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    Grad-CAM++ only differs from [GradCAM][aucmedi.xai.methods.gradcam.GradCAM] in how the
    gradient is weighted, which is why the identification of the output layer as well as the
    feature map computation are inherited.

    ??? abstract "Reference - Implementation"
        Author: Jacob Gil <br>
        GitHub Profile: [https://github.com/jacobgil](https://github.com/jacobgil) <br>
        Date: 2021 <br>
        [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) <br>

    ??? abstract "Reference - Publication"
        Aditya Chattopadhay; Anirban Sarkar; Prantik Howlader; Vineeth N Balasubramanian. 07 May 2018.
        Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks.
        <br>
        [https://ieeexplore.ieee.org/document/8354201](https://ieeexplore.ieee.org/document/8354201)

    This class provides functionality for running the compute_heatmap function,
    which computes a Grad-CAM++ heatmap for an image with a model.
    """
    def __init__(self, model, layerName=None):
        """ Initialization function for creating a Grad-CAM++ as XAI Method object.

        Args:
            model (nn.Module):              PyTorch model object.
            layerName (str):                   Layer name of the convolutional layer for heatmap computation.
        """
        super().__init__(model, layerName)

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Grad-CAM++ heatmap for a provided image and for specific classification outcome.

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
            heatmap (numpy.ndarray):            Computed Grad-CAM++ for provided image.
        """
        # Obtain feature map of the last conv layer and its gradient
        conv_out, grads = self.compute_feature_gradient(image, class_index)

        # Identify spatial axis (keep batch & channel axis)
        spatial_axis = tuple(range(2, grads.dim()))
        # Derive the second and third order derivative from the first one. Following the
        # publication, the class score is modeled as an exponential, which reduces the
        # higher order derivatives to powers of the first order gradient. Backpropagating
        # them instead would return zeros for the piecewise linear ReLU architectures.
        conv_second_grad = grads.pow(2)
        conv_third_grad = conv_second_grad * grads

        # Normalize constants
        global_sum = conv_out.sum(dim=spatial_axis, keepdim=True)
        alpha_denom = conv_second_grad * 2.0 + conv_third_grad * global_sum
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom,
                                  torch.full_like(alpha_denom, eps))
        alphas = conv_second_grad / alpha_denom

        # Deep Linearization weighting
        weights = torch.clamp(grads, min=0.0)
        deep_linearization_weights = (weights * alphas).sum(dim=spatial_axis)
        # Normalize gradients via "importance"
        conv_out = conv_out[0]
        deep_linearization_weights = deep_linearization_weights[0].view(
            -1, *([1] * (conv_out.dim() - 1)))
        heatmap = (conv_out * deep_linearization_weights).sum(dim=0)
        heatmap = heatmap.detach().cpu().numpy()

        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom

        # Return the resulting heatmap
        return heatmap
