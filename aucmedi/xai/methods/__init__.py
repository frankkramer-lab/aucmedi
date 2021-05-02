#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
# XAI method dictionary
xai_dict = {
    "gradcam": GradCAM,
    "GradCAM": GradCAM,
    "gradcam++": GradCAMpp,
    "GradCAM++": GradCAMpp,
    "GradCAMpp": GradCAMpp,
    "saliency": SaliencyMap,
    "SaliencyMap": SaliencyMap,
    "guidedbackprop": GuidedBackpropagation,
    "GuidedBackpropagation": GuidedBackpropagation
}
