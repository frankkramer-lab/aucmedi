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
# External Libraries
import numpy as np
from lime import lime_image
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                  LIME: Pro Features                 #
#-----------------------------------------------------#
class LimePro(XAImethod_Base):
    """ XAI Method for LIME Pro.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Lime: Explaining the predictions of any machine learning classifier <br>
        GitHub: https://github.com/marcotcr/lime <br>

    ??? abstract "Reference - Publication"
        Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. 9 Aug 2016.
        "Why Should I Trust You?": Explaining the Predictions of Any Classifier
        <br>
        https://arxiv.org/abs/1602.04938

    This class provides functionality for running the compute_heatmap function,
    which computes a Lime Pro Map for an image with a model.
    """
    def __init__(self, model, layerName=None, num_samples=1000):
        """ Initialization function for creating a Lime Pro Map as XAI Method object.

        Args:
            model (keras.model):            Keras model object.
            layerName (str):                Not required in LIME Pro/Con Maps, but defined by Abstract Base Class.
            num_samples (int):              Number of iterations for LIME instance explanation.
        """
        # Cache class parameters
        self.model = model
        self.num_samples = num_samples

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Lime Pro Map for a provided image and for specific classification outcome.

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
            heatmap (numpy.ndarray):            Computed Lime Pro Map for provided image.
        """
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
