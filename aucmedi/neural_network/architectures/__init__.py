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
# Abstract Base Class for Architectures
from aucmedi.neural_network.architectures.arch_base import Architecture_Base

#-----------------------------------------------------#
#       Access Functions to Architecture Classes      #
#-----------------------------------------------------#
# Initialize combined architecture_dict for image & volume architectures
architecture_dict = {}

# Add image architectures to architecture_dict
from aucmedi.neural_network.architectures.image import architecture_dict as arch_image
for arch in arch_image:
    architecture_dict["2D." + arch] = arch_image[arch]

# Add volume architectures to architecture_dict

#-----------------------------------------------------#
#       Meta Information of Architecture Classes      #
#-----------------------------------------------------#
# Initialize combined supported_standardize_mode for image & volume architectures
supported_standardize_mode = {}

# Add image architectures to supported_standardize_mode
from aucmedi.neural_network.architectures.image import supported_standardize_mode as modes_image
for m in modes_image:
    supported_standardize_mode["2D." + m] = modes_image[m]

# Add volume architectures to supported_standardize_mode
