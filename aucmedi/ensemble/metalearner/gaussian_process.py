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
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np
# Internal libraries/scripts
from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base

#-----------------------------------------------------#
#            Metalearner: Gaussian Process            #
#-----------------------------------------------------#
class GaussianProcess(Metalearner_Base):
    """ A Gaussian Process based Metalearner.

    This class should be passed to an ensemble function/class like Stacking for combining predictions.

    !!! warning
        Can only be utilized for binary and multi-class tasks.

        Does not work on multi-label annotations!

    ???+ abstract "Reference - Implementation"
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html

        Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
        https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self):
        self.model = GaussianProcessClassifier(random_state=0,
                                               multi_class="one_vs_rest")

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    def train(self, x, y):
        # Preprocess to sparse encoding
        y = np.argmax(y, axis=-1)
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
