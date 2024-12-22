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
# Python Standard Library

# Third Party Libraries
import matplotlib.cm as cm
import numpy as np
from PIL import Image


#-----------------------------------------------------#
#                   Image Visualizer                  #
#-----------------------------------------------------#
def visualize_array(array, out_path=None):
    """ Simple wrapper function to visualize a NumPy matrix as image via PIL.

    ???+ info
        NumPy array shape has to be (x, y, channel) like this: (224, 224, 3)

    Args:
        array (numpy.ndarray):          NumPy matrix containing an image.
        out_path (str):                 Path in which image is stored (else live output).
    """
    # Ensure integer intensity values
    array = np.uint8(array)
    # Remove channel axis if grayscale
    if array.shape[-1] == 1: array = np.reshape(array, array.shape[:-1])
    # Convert array to PIL image
    image = Image.fromarray(array)
    # Visualize or store image
    if out_path is None:
        image.show()
    else:
        image.save(out_path)


#-----------------------------------------------------#
#               XAI Heatmap Visualizer                #
#-----------------------------------------------------#
def visualize_heatmap(image, heatmap, out_path=None, alpha=0.4):
    """ Simple wrapper function to visualize a heatmap encoded as NumPy matrix with a
        [0-1] range as image via matplotlib and PILLOW.

    ??? abstract "Reference - Implementation"
        Author: François Chollet <br>
        Date: April 26, 2020 <br>
        https://keras.io/examples/vision/grad_cam/ <br>

    Args:
        image (numpy.ndarray):          NumPy matrix containing an image.
        heatmap (numpy.ndarray):        NumPy matrix containing a XAI heatmap.
        out_path (str):                 Path in which image is stored (else live output).
        alpha (float):                  Transparency value for heatmap overlap on image (range: [0-1]).
    """
    # If image is grayscale, convert to RGB
    if image.shape[-1] == 1: image = np.concatenate((image,)*3, axis=-1)
    # Rescale heatmap to grayscale range
    heatmap = np.uint8(heatmap * 255)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:,:3]
    jet_heatmap = jet_colors[heatmap] * 255
    # Superimpose the heatmap on original image
    si_img = jet_heatmap * alpha + (1-alpha) * image
    # Convert array to PIL image
    si_img = si_img.astype(np.uint8)
    pil_img = Image.fromarray(si_img)
    # Visualize or store image
    if out_path is None:
        pil_img.show()
    else:
        pil_img.save(out_path)
