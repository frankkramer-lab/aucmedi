#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
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
import pandas as pd
# Internal libraries/scripts
from ensmic.ensemble.abstract_elm import Abstract_Ensemble

#-----------------------------------------------------#
#              ELM: Majority Vote - Hard              #
#-----------------------------------------------------#
""" Ensemble Learning approach via Hard Majority Vote.

Methods:
    __init__                Initialize Ensemble Learning Method.
    training:               Fit Ensemble Learning Method on validate-ensemble.
    prediction:             Utilize Ensemble Learning Method for test dataset.
    dump:                   Save (fitted) model to disk.
    load:                   Load (fitted) model from disk.
"""
class ELM_MajorityVote_Hard(Abstract_Ensemble):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_classes):
        # No hyperparameter adjustment required for this method, therefore skip
        pass

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    def training(self, train_x, train_y):
        # No training required for this method, therefore skip
        pass

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    def prediction(self, data):
        # Split data columns into multi level structure based on architecutre
        data.columns = data.columns.str.split('_', expand=True)
        # Identify argmax (vote) for each architecutre and cache n-architectures
        data = data.groupby(level=0, axis=1).idxmax(axis=1)
        n_architectures = len(data.columns)
        data = data.apply(lambda entry: [tup[1] for tup in entry])
        # Sum up votes of all architectures
        data = data.apply(pd.Series.value_counts, axis=1).fillna(0)
        # Compute Class Probabilities
        pred = data.divide(n_architectures)
        # Return prediction
        return pred.to_numpy()

    #---------------------------------------------#
    #              Dump Model to Disk             #
    #---------------------------------------------#
    def dump(self, path):
        # No model infrastructure required for this method, therefore skip
        pass

    #---------------------------------------------#
    #             Load Model from Disk            #
    #---------------------------------------------#
    def load(self, path):
        # No model infrastructure required for this method, therefore skip
        pass
