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
# Import XAI methods
from aucmedi.xai.methods.gradcam import GradCAM
from aucmedi.xai.methods.gradcam_pp import GradCAMpp
from aucmedi.xai.methods.saliency import SaliencyMap
from aucmedi.xai.methods.guided_backprop import GuidedBackpropagation
from aucmedi.xai.methods.integrated_gradients import IntegratedGradients
from aucmedi.xai.methods.gradcam_guided import GuidedGradCAM
from aucmedi.xai.methods.occlusion_sensitivity import OcclusionSensitivity
from aucmedi.xai.methods.lime_pro import LimePro
from aucmedi.xai.methods.lime_con import LimeCon

#-----------------------------------------------------#
#                XAI method dictionary                #
#-----------------------------------------------------#
xai_dict = {
    "gradcam": GradCAM,
    "GradCAM": GradCAM,
    "gc": GradCAM,
    "gradcam++": GradCAMpp,
    "GradCAM++": GradCAMpp,
    "GradCAMpp": GradCAMpp,
    "gc++": GradCAMpp,
    "GuidedGradCAM": GuidedGradCAM,
    "ggc": GuidedGradCAM,
    "saliency": SaliencyMap,
    "SaliencyMap": SaliencyMap,
    "sm": SaliencyMap,
    "guidedbackprop": GuidedBackpropagation,
    "GuidedBackpropagation": GuidedBackpropagation,
    "gb": GuidedBackpropagation,
    "IntegratedGradients": IntegratedGradients,
    "ig":IntegratedGradients,
    "OcclusionSensitivity": OcclusionSensitivity,
    "os": OcclusionSensitivity,
    "LimePro": LimePro,
    "lp": LimePro,
    "LimeCon": LimeCon,
    "lc": LimeCon
}
""" Dictionary of implemented XAI Methods in AUCMEDI.

    A key (str) or an initialized XAI Method can be passed to the [aucmedi.xai.decoder.xai_decoder][] function as method parameter.

    ???+ example "Example"
        ```python
        # Select desired XAI Methods
        xai_list = ["gradcam", "gc++", OcclusionSensitivity(model), xai_dict["LimePro"](model), "lc"]

        # Iterate over each method
        for m in xai_list:
            # Compute XAI heatmaps with method m
            heatmaps = xai_decoder(datagen, model, preds, method=m)
        ```

    XAI Methods are based on the abstract base class [aucmedi.xai.methods.xai_base][].
"""
