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
import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# Internal libraries/scripts
from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base

#-----------------------------------------------------#
#             Metalearner: Decision Tree              #
#-----------------------------------------------------#
class Decision_Tree(Metalearner_Base):
    """ A Decision Tree based Metalearner.

    This class should be passed to a Ensemble function like Stacking for combining predictions.

    !!! info
        Can be utilized for binary, multi-class and multi-label tasks.
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=0)

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    def train(self, x, y):
        # Train model
        self.model = self.model.fit(x, y)

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    def predict(self, data):
        # Compute prediction probabilities via fitted model
        pred = self.model.predict_proba(data)
        # Return results as NumPy array
        return pred

    #---------------------------------------------#
    #              Dump Model to Disk             #
    #---------------------------------------------#
    def dump(self, path):
        # Dump model to disk via pickle
        with open(path, "wb") as pickle_writer:
            pickle.dump(self.model, pickle_writer)

    #---------------------------------------------#
    #             Load Model from Disk            #
    #---------------------------------------------#
    def load(self, path):
        # Load model from disk via pickle
        with open(path, "rb") as pickle_reader:
            self.model = pickle.load(pickle_reader)
