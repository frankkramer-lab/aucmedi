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
# XAI method dictionary
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
    "OcclusionSensitivity":OcclusionSensitivity,
    "os":OcclusionSensitivity,
    "LimePro":LimePro,
    "lp":LimePro,
    "LimeCon":LimeCon,
    "lc":LimeCon
}
