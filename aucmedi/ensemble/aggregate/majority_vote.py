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
# Python Standard Library

# Third Party Libraries
import numpy as np

# Internal Libraries
from aucmedi.ensemble.aggregate.agg_base import Aggregate_Base


#-----------------------------------------------------#
#               Aggregate: Majority Vote              #
#-----------------------------------------------------#
class MajorityVote(Aggregate_Base):
    """ Aggregate function based on majority vote.

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
        # Count votes
        votes = np.argmax(preds, axis=1)
        # Identify majority
        majority_vote = np.argmax(np.bincount(votes))
        # Create prediction based on majority vote
        pred = np.zeros((preds.shape[1]))
        pred[majority_vote] = 1
        # Return prediction
        return pred
