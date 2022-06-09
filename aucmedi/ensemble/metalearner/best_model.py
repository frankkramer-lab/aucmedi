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
from sklearn.metrics import roc_auc_score
import numpy as np
# Internal libraries/scripts
from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base

#-----------------------------------------------------#
#              Metalearner: Best Model                #
#-----------------------------------------------------#
class BestModel(Metalearner_Base):
    """ A Best Model based Metalearner.

    This class should be passed to an ensemble function/class like Stacking for combining predictions.

    This Metalearner computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    for each model, and simply utilizes only the predictions of the best scoring model.

    !!! info
        Can be utilized for binary, multi-class and multi-label tasks.
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self):
        self.model = {}

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    def train(self, x, y):
        # Identify number of models and classes
        n_classes = y.shape[1]
        n_models = int(x.shape[1] / n_classes)
        # Preprocess data input
        data = np.reshape(x, (x.shape[0], n_models, n_classes))

        # Compute AUC scores and store them to cache
        for m in range(n_models):
            pred = data[:,m,:]
            score = roc_auc_score(y, pred, average="macro")
            self.model["nn_" + str(m)] = score

        # Identify best model and store results to cache
        best_model = max(self.model, key=self.model.get)
        self.model["best_model"] = best_model
        self.model["n_classes"] = n_classes
        self.model["n_models"] = n_models

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    def predict(self, data):
        # Preprocess data input
        preds = np.reshape(data, (data.shape[0],
                                  self.model["n_models"],
                                  self.model["n_classes"]))

        # Obtain prediction probabilities of best model
        m = int(self.model["best_model"].split("_")[-1])
        pred_best = preds[:,m,:]
        # Return results
        return pred_best

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
