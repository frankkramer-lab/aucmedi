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
#         Abstract Base Class for Subfunctions        #
#-----------------------------------------------------#
class Subfunction_Base(ABC):
    """ An abstract base class for a Subfunction class.

    A child of this ABC can be used as a [Subfunction][aucmedi.data_processing.subfunctions]
    and be passed to a [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    This class provides functionality for running the transform function,
    which preprocesses an image during the processing (batch preparation) of the DataGenerator.

    ???+ example "Create a custom Subfunction"
        ```python
        from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

        class My_custom_Subfunction(Subfunction_Base):
            def __init__(self):                 # you can pass here class variables
                pass

            def transform(self, image):
                new_image = image + 1.0         # do some operation
                return new_image                # return modified image
        ```

    ???+ info "Required Functions"
        | Function            | Description                                |
        | ------------------- | ------------------------------------------ |
        | `__init__()`        | Object creation function.                  |
        | `transform()`       | Transform the image.                       |

    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    @abstractmethod
    def __init__(self):
        """ Functions which will be called during the Subfunction object creation.

        ```
        __init__(model, layerName=None)
        ```

        This function can be used to pass variables and options in the Subfunction instance.
        The are no mandatory required parameters for the initialization.
        """
        pass
    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    @abstractmethod
    def transform(self, image):
        """ Transform the image according to the subfunction during preprocessing (training + prediction).

        It is required to return the transformed image object (as NumPy array).

        Args:
            image (numpy.ndarray):      Image encoded as NumPy matrix with 1 or 3 channels. (e.g. 224x224x3)

        Returns:
            image (numpy.ndarray):      Transformed image encoded as NumPy matrix with 1 or 3 channels. (e.g. 224x224x3)
        """
        return image
