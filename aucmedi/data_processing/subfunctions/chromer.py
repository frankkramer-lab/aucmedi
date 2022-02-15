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
import numpy as np
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#              Subfunction class: Chromer             #
#-----------------------------------------------------#
""" A chromer Subfunction class which can be used for transforming:
        grayscale ->  RGB
        RGB       ->  grayscale

    Possible target formats : ["grayscale", "rgb"]

    Typical use case is converting a grayscale to RGB in order to utilize
    transfer learning weights on ImageNet.

Methods:
    __init__                Object creation function.
    transform:              Apply chromer.
"""
class Chromer(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, target="rgb"):
        # Verify target format
        if target not in ["grayscale", "rgb"]:
            raise ValueError("Unknown target format for Chromer Subfunction",
                             target, "Possibles target formats are: ['grayscale', 'rgb']")
        # Cache target format
        self.target = target

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Verify that image is in correct format
        if self.target == "rgb" and (image.shape[-1] != 1 or \
                                     np.max(image) > 255 or \
                                     np.min(image) < 0):
            raise ValueError("Subfunction Chromer: Image is not in grayscale format!",
                             "Ensure that it is grayscale normalized and has",
                             "a single channel.")
        elif self.target == "grayscale" and (image.shape[-1] != 3 or \
                                             np.max(image) > 255 or \
                                             np.min(image) < 0):
            raise ValueError("Subfunction Chromer: Image is not in RGB format!",
                             "Ensure that it is normalized [0,255] and has 3 channels.")
        # Run grayscale -> RGB
        if self.target == "rgb":
            image_chromed = np.concatenate((image,)*3, axis=-1)
        # Run RGB -> grayscale
        else:
            # Get color intensity values
            r = np.take(image, indices=0, axis=-1)
            g = np.take(image, indices=1, axis=-1)
            b = np.take(image, indices=2, axis=-1)
            # Compute grayscale image
            image_chromed = 0.299 * r + 0.587 * g + 0.114 * b
            # Add channel axis back
            image_chromed = np.expand_dims(image_chromed, axis=-1)
        # Return chromed image
        return image_chromed


# convert grayscale to RGB
# convert RGB to grayscale
