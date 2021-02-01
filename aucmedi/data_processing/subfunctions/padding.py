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
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" A Padding Subfunction class which pads an images according to a desired shape.

    Standard application is to square images to keep original aspect ratio.
    If another mode as "square" is selected, than a shape and NumPy pad mode is required!

    Shape should be defined as tuple with x and y size:
    Padding(shape=(224, 224))

    Padding is done via NumPy pad function which uses can be called with different
    modes like "edge" or "minimum".
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html

Methods:
    __init__                Object creation function
    transform:              Pad an image input according to desired shape.
"""
class Padding(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="square", shape=None):
        self.shape = shape
        self.mode = mode

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Identify new shape
        if self.mode == "square":
            max_axis = max(image.shape[:-1])
            new_shape = [max_axis, max_axis]
        else:
            new_shape = [max(self.shape[i],image.shape[i]) for i in range(0, 2)]
        # Compute padding width
        ## Code inspiration from: https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py
        ## Leave a star for them if you are reading this. The MIC-DKFZ is doing some great work ;)
        difference = new_shape - np.asarray(image.shape[0:-1])
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = list([list(i) for i in zip(pad_below, pad_above)]) + [[0, 0]]
        # Identify correct NumPy pad mode
        if self.mode == "square" : pad_mode = "constant"
        else : pad_mode = self.mode
        # Perform padding into desired shape
        image_padded = np.pad(image, pad_list, mode=pad_mode)
        # Return padded image
        return image_padded
