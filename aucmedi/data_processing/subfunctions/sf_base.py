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
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#         Abstract Base Class for Subfunctions        #
#-----------------------------------------------------#
""" An abstract base class for a preprocessing Subfunction class.

Methods:
    __init__                Object creation function.
    transform:              Transform the imaging data.
"""
class Subfunction_Base(ABC):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Functions which will be called during the Subfunction object creation.
        This function can be used to pass variables and options in the Subfunction instance.
        The are no mandatory required parameters for the initialization.

        Parameter:
            None
        Return:
            None
    """
    @abstractmethod
    def __init__(self):
        pass
    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    """ Transform the image according to the subfunction during preprocessing (training + prediction).
        It is required to return the transformed image object (as NumPy array).
        It is possible to pass configurations through the initialization function for this class.

        Parameter:
            image (Numpy Array):        Image encoded as NumPy matrix with 1 or 3 channels.
        Return:
            image (Numpy Array):        Transformed image encoded as NumPy matrix with 1 or 3 channels.
    """
    @abstractmethod
    def transform(self, image):
        return image
