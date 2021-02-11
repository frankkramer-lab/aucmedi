#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
from aucmedi.ensembler.aggregate.agg_base import Aggregate_Base

#-----------------------------------------------------#
#            Aggregate: Averaging by Mean             #
#-----------------------------------------------------#
""" Aggregate function based on averaging via mean.

    Methods:
        __init__:               Object creation function.
        aggregate:              Merge multiple class predictions into a single prediction.
"""

class Averaging_Mean(Aggregate_Base):
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
        # Merge predictions via mean
        pred = np.mean(preds, axis=0)
        # Return merged prediction
        return pred
