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
# External libraries
import os
import numpy as np
from PIL import Image

#-----------------------------------------------------#
#             Image Loader for AUCMEDI IO             #
#-----------------------------------------------------#
""" Image Loader for image loading within the AUCMEDI pipeline.

    Arguments:
        sample (String):                Sample name/index of an image.
        path_imagedir (String):         Path to the directory containing the images.
        image_format (String):          Image format to add at the end of the sample index for image loading.
        grayscale (Boolean):            Boolean, whether images are grayscale or RGB.
        kwargs (Dictionary):            Additional parameters for the sample loader.
"""
def image_loader(sample, path_imagedir, image_format=None, grayscale=False,
                 **kwargs):
    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    path_img = os.path.join(path_imagedir, img_file)
    # Load image via the PIL package
    img_raw = Image.open(path_img)
    # Convert image to grayscale or rgb
    if grayscale : img_converted = img_raw.convert('LA')
    else : img_converted = img_raw.convert('RGB')
    # Convert image to NumPy
    img = np.asarray(img_converted)
    # Perform additional preprocessing if grayscale image
    if grayscale:
        # Remove maximum value and keep only intensity
        img = img[:,:,0]
        # Reshape image to create a single channel
        img = np.reshape(img, img.shape + (1,))
    # Return image
    return img
