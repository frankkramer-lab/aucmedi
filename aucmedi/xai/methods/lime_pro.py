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
#                      REFERENCE:                     #
# Lime: Explaining the predictions of any machine     #
# learning classifier                                 #
# GitHub: https://github.com/marcotcr/lime            #
#-----------------------------------------------------#
#                     PUBLICATION:                    #
#                     9 Aug 2016.                     #
#      "Why Should I Trust You?": Explaining the      #
#            Predictions of Any Classifier            #
#  Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin #
#          https://arxiv.org/abs/1602.04938           #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External Libraries
import numpy as np
from lime import lime_image
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                  LIME: Pro Features                 #
#-----------------------------------------------------#
class LimePro(XAImethod_Base):
    """ Initialization function for creating a Lime Pro Map as XAI Method object.
    Normally, this class is used internally in the xai_decoder function in the AUCMEDI XAI module.

    This class provides functionality for running the compute_heatmap function,
    which computes a Lime Pro Map for an image with a model.

    Args:
        model (Keras Model):               Keras model object.
        layerName (String):                Not required in Lime Pro Maps, but defined by Abstract Base Class.
        num_samples (Integer):             Number of iterations for LIME instance explanation.
    """
    def __init__(self, model, layerName=None, num_samples=1000):
        # Cache class parameters
        self.model = model
        self.num_samples = num_samples

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    """ Core function for computing the Lime Pro Map for a provided image and for specific classification outcome.
    The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

    Be aware that the image has to be provided in batch format.

    Args:
        image (NumPy Array):                Image matrix encoded as NumPy Array (provided as one-element batch).
        class_index (Integer):              Classification index for which the heatmap should be computed.
        eps (Float):                        Epsilon for rounding.
    """
    def compute_heatmap(self, image, class_index, eps=1e-8):
        # Initialize LIME explainer
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image[0].astype("double"),
                                self.model.predict, hide_color=0,
                                labels=(class_index,),
                                num_samples=self.num_samples)
        # Obtain PRO explanation mask
        temp, mask = explanation.get_image_and_mask(class_index, hide_rest=True,
                                positive_only=True, negative_only=False)
        heatmap = mask
        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        # Return the resulting heatmap
        return heatmap
