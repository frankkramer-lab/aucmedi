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
# Import aggregate functions
from aucmedi.ensembler.aggregate.averaging_mean import Averaging_Mean
from aucmedi.ensembler.aggregate.averaging_median import Averaging_Median
from aucmedi.ensembler.aggregate.majority_vote import Majority_Vote
from aucmedi.ensembler.aggregate.softmax import Softmax

#-----------------------------------------------------#
#       Access Functions to Aggregate Functions       #
#-----------------------------------------------------#
aggregate_dict = {"mean": Averaging_Mean,
                  "median": Averaging_Median,
                  "majority_vote": Majority_Vote,
                  "softmax": Softmax}
