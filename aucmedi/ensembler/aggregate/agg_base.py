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
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#         Abstract Base Class for Aggregation         #
#-----------------------------------------------------#
class Aggregate_Base(ABC):
    """ An abstract base class for a Aggregation class.

    ```
    Augmented predictions encoded in a NumPy Matrix with shape (N_cycles, N_classes).
    Example: [[0.5, 0.4, 0.1],
              [0.4, 0.3, 0.3],
              [0.5, 0.2, 0.3]]
    -> shape (3, 3)

    Merged prediction encoded in a NumPy Matrix with shape (1, N_classes).
    Example: [[0.4, 0.3, 0.3]]
    -> shape (1, 3)
    ```

    ???+ example "Create a custom Aggregation class"
        ```python
        from aucmedi.ensembler.aggregate.agg_base import Aggregate_Base

        class My_custom_Aggregate(Aggregate_Base):
            def __init__(self):                 # you can pass here class variables
                pass

            def aggregate(self, preds):
                preds_combined = np.mean(preds, axis=0)     # do some combination operation
                return preds_combined                       # return combined predictions
        ```

    !!! info "Required Functions"
        | Function            | Description                                                |
        | ------------------- | ---------------------------------------------------------- |
        | `__init__()`        | Object creation function.                                  |
        | `aggregate()`       | Merge multiple class predictions into a single prediction. |
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    @abstractmethod
    def __init__(self):
        """ Initialization function which will be called during the Aggregation object creation.

        This function can be used to pass variables and options in the Aggregation instance.
        The are no mandatory required parameters for the initialization.
        """
        pass
    #---------------------------------------------#
    #                  Aggregate                  #
    #---------------------------------------------#
    @abstractmethod
    def aggregate(self, preds):
        """ Aggregate the image according to the subfunction during preprocessing (training + prediction).

        It is required to return the merged predictions (as NumPy matrix).
        It is possible to pass configurations through the initialization function for this class.

        Args:
            preds (numpy.ndarray):      Augmented predictions encoded in a NumPy Matrix with shape (N_cycles, N_classes).
        Returns:
            pred (numpy.ndarray):       Merged prediction encoded in a NumPy Matrix with shape (1, N_classes).
        """
        return pred
