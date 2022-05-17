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
# Internal libraries
from aucmedi import DataGenerator
from aucmedi.ensemble.aggregate import aggregate_dict

#-----------------------------------------------------#
#              Ensemble Learning: Bagging             #
#-----------------------------------------------------#
class Bagging:
    """ A Bagging class providing functionality for cross-validation based ensemble learning.

    """
    def __init__(self, model, k_fold=3):
        """ Initialization function for creating a Bagging object.
        """
        # Cache class variables
        self.model_template = model
        self.k_fold = k_fold
        self.model_list = []

        # Create k models based on template
        for i in range(k_fold):



    def train(self, training_generator, epochs=20,
              iterations=None, callbacks=[], class_weights=None,
              transfer_learning=False):
        # apply cross-validaton
        pass
        # for loop
            # model.training
        # return combined history object


    def predict(self, prediction_generator, aggregate=""):
        pass
        # for loop
            # model.predict
        # aggregate
        # return


    # Dump model to file
    def dump(self, file_path):
        """ Store model to disk.

        Recommended to utilize the file format ".hdf5".

        Args:
            file_path (str):    Path to store the model on disk.
        """
        self.model.save(file_path)


    # Load model from file
    def load(self, file_path, custom_objects={}):
        """ Load neural network model and its weights from a file.

        After loading, the model will be compiled.

        If loading a model in ".hdf5" format, it is not necessary to define any custom_objects.

        Args:
            file_path (str):            Input path, from which the model will be loaded.
            custom_objects (dict):      Dictionary of custom objects for compiling
                                        (e.g. non-TensorFlow based loss functions or architectures).
        """
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss=self.loss, metrics=self.metrics)
