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
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#         Abstract Base Class for Metalearner         #
#-----------------------------------------------------#
class Metalearner_Base(ABC):
    """ An abstract base class for a Metalearner class.

    Metalearner are similar to [Aggregate functions][aucmedi.ensemble.aggregate],
    however Metalearners are models which require fitting before usage.

    Metalearners are utilized in [Stacking][aucmedi.ensemble.stacking] pipelines.

    A Metalearner act as a combiner algorithm which is trained to make a final prediction
    using predictions of other algorithms (`NeuralNetwork`) as inputs.

    ```
    Assembled predictions encoded in a NumPy matrix with shape (N_models, N_classes).
    Example: [[0.5, 0.4, 0.1],
              [0.4, 0.3, 0.3],
              [0.5, 0.2, 0.3]]
    -> shape (3, 3)

    Merged prediction encoded in a NumPy matrix with shape (1, N_classes).
    Example: [[0.4, 0.3, 0.3]]
    -> shape (1, 3)
    ```

    !!! info "Required Functions"
        | Function            | Description                                                |
        | ------------------- | ---------------------------------------------------------- |
        | `__init__()`        | Object creation function.                                  |
        | `training()`        | Fit Metalearner model.                                     |
        | `prediction()`      | Merge multiple class predictions into a single prediction. |
        | `dump()`            | Store Metalearner model to disk.                           |
        | `load()`            | Load Metalearner model from disk.                          |
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    @abstractmethod
    def __init__(self):
        """ Initialization function which will be called during the Metalearner object creation.

        This function can be used to pass variables and options in the Metalearner instance.
        There are no mandatory parameters for the initialization.
        """
        pass

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    @abstractmethod
    def train(self, x, y):
        """ Training function to fit the Metalearner model.

        Args:
            x (numpy.ndarray):          Assembled prediction dataset encoded in a NumPy matrix with shape (N_samples, N_classes*N_models).
            y (numpy.ndarray):          Classification list with One-Hot Encoding. Provided by
                                        [input_interface][aucmedi.data_processing.io_data.input_interface].
        """
        pass

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    @abstractmethod
    def predict(self, data):
        """ Merge multiple predictions for a sample into a single prediction.

        It is required to return the merged predictions (as NumPy matrix).
        It is possible to pass configurations through the initialization function for this class.

        Args:
            data (numpy.ndarray):       Assembled predictions encoded in a NumPy matrix with shape (N_models, N_classes).
        Returns:
            pred (numpy.ndarray):       Merged prediction encoded in a NumPy matrix with shape (1, N_classes).
        """
        pass

    #---------------------------------------------#
    #              Dump Model to Disk             #
    #---------------------------------------------#
    @abstractmethod
    def dump(self, path):
        """ Store Metalearner model to disk.

        Args:
            path (str):                 Path to store the model on disk.
        """
        pass

    #---------------------------------------------#
    #             Load Model from Disk            #
    #---------------------------------------------#
    @abstractmethod
    def load(self, path):
        """ Load Metalearner model and its weights from a file.

        Args:
            path (str):                 Input path from which the model will be loaded.
        """
        pass
