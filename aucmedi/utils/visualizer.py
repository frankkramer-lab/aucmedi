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
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from PIL import Image

#-----------------------------------------------------#
#                   Image Visualizer                  #
#-----------------------------------------------------#
""" Simple wrapper function to visualize a NumPy matrix as image via PIL.

    NumPy array shape has to be (x, y, channel) like this: (224, 224, 3)

    Arguments:
        array (NumPy matrix):           NumPy matrix containing an image.
        out_path (String):              Path in which image is stored (else live output).
"""
def visualize_array(array, out_path=None):
    # Ensure integer intensity values
    array = np.uint8(array)
    # Remove channel axis if grayscale
    if array.shape[-1] == 1 : array = np.reshape(array, array.shape[:-1])
    # Convert array to PIL image
    image = Image.fromarray(array)
    # Visualize or store image
    if out_path is None : image.show()
    else : image.save(out_path)
