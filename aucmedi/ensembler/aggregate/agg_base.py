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
#         Abstract Base Class for Aggregation         #
#-----------------------------------------------------#
""" An abstract base class for a Aggregation class.

    Augmented predictions encoded in a NumPy Matrix with shape (N_cycles, N_classes).
    Example: [[0.5, 0.4, 0.1],
              [0.4, 0.3, 0.3],
              [0.5, 0.2, 0.3]]
    -> shape (3, 3)

    Merged prediction encoded in a NumPy Matrix with shape (1, N_classes).
    Example: [[0.4, 0.3, 0.3]]
    -> shape (1, 3)

    Methods:
        __init__:               Object creation function.
        aggregate:              Merge multiple class predictions into a single prediction.
"""
class Aggregate_Base(ABC):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Functions which will be called during the Aggregation object creation.
        This function can be used to pass variables and options in the Aggregation instance.
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
    #                  Aggregate                  #
    #---------------------------------------------#
    """ Aggregate the image according to the subfunction during preprocessing (training + prediction).
        It is required to return the merged predictions (as NumPy matrix).
        It is possible to pass configurations through the initialization function for this class.

        Parameter:
            preds (Numpy Matrix):       Augmented predictions encoded in a NumPy Matrix with shape (N_cycles, N_classes).
        Return:
            pred (Numpy Matrix):        Merged prediction encoded in a NumPy Matrix with shape (1, N_classes).
    """
    @abstractmethod
    def aggregate(self, preds):
        return pred
