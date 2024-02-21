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
import numpy as np
# Internal libraries/scripts
from aucmedi.ensemble.aggregate.agg_base import Aggregate_Base

#-----------------------------------------------------#
#               Aggregate: Global Argmax              #
#-----------------------------------------------------#
class GlobalArgmax(Aggregate_Base):
    """ Aggregate function based on Global Argmax.

    This class should be passed to an ensemble function/class for combining predictions.
    """
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self):
        # No hyperparameter adjustment required for this method, therefore skip
        pass

    #---------------------------------------------#
    #                  Aggregate                  #
    #---------------------------------------------#
    def aggregate(self, preds):
        # Identify global argmax
        max = np.amax(preds)
        argmax_flatten = np.argmax(preds)
        argmax = np.unravel_index(argmax_flatten, preds.shape)[-1]

        # Compute prediction by global argmax and equally distributed remaining
        # probability for other classes
        prob_remaining = np.divide(1-max, preds.shape[1]-1)
        pred = np.full((preds.shape[1],), fill_value=prob_remaining)
        pred[argmax] = max

        # Return prediction
        return pred
