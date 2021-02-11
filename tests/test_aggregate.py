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
#External libraries
import unittest
import numpy as np
#Internal libraries
from aucmedi.ensembler.aggregate import *

#-----------------------------------------------------#
#             Unittest: Aggregate Functions           #
#-----------------------------------------------------#
class AggregateTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create data
        self.pred_data = np.random.rand(50, 10)

    #-------------------------------------------------#
    #           Aggregate: Averaging by Mean          #
    #-------------------------------------------------#
    def test_Aggregate_Mean(self):
        agg_func = Averaging_Mean()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (10,)))

    #-------------------------------------------------#
    #          Aggregate: Averaging by Median         #
    #-------------------------------------------------#
    def test_Aggregate_Median(self):
        agg_func = Averaging_Median()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (10,)))

    #-------------------------------------------------#
    #         Aggregate: Majority Vote (Hard)         #
    #-------------------------------------------------#
    def test_Aggregate_MajorityVote(self):
        agg_func = Majority_Vote()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (10,)))

    #-------------------------------------------------#
    #                Aggregate: Softmax               #
    #-------------------------------------------------#
    def test_Aggregate_Softmax(self):
        agg_func = Softmax()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (10,)))
