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
# External libraries
import numpy as np
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#               Subfunction class: Clip               #
#-----------------------------------------------------#
class Clip(Subfunction_Base):
    """ A Subfunction class which which can be used for clipping intensity pixel
        values on a certain range.

    Typical use case is clipping Hounsfield Units (HU) in CT scans for focusing
    on tissue types of interest.
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, min=None, max=None, per_channel=False):
        """ Initialization function for creating a Clip Subfunction which can be passed to a
            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            min (float or int or list):     Desired minimum value for clipping (if `None`, no lower limit is applied).
                                            Also possible to pass a list of minimum values if `per_channel=True`.
            max (float or int or list):     Desired maximum value for clipping (if `None`, no upper limit is applied).
                                            Also possible to pass a list of maximum values if `per_channel=True`.
            per_channel (bool):             Option if clipping should be applied per channel with different clipping ranges.
        """
        self.min = min
        self.max = max
        self.per_channel = per_channel

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform clipping on all channels
        if not self.per_channel:
            image_clipped = np.clip(image, a_min=self.min, a_max=self.max)
        # Perform clipping on each channel
        else:
            image_clipped = image.copy()
            for c in range(0, image.shape[-1]):
                # Identify minimum to clip
                if self.min is not None and \
                    type(self.min) in [list, tuple, np.ndarray]:
                    min = self.min[c]
                else : min = self.min
                # Identify maximum to clip
                if self.max is not None and \
                    type(self.max) in [list, tuple, np.ndarray]:
                    max = self.max[c]
                else : max = self.max
                # Perform clipping
                image_clipped[..., c] = np.clip(image[...,c], 
                                                a_min=min, 
                                                a_max=max)
        # Return clipped image
        return image_clipped
