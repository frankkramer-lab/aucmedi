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
#External libraries
import unittest
import numpy as np
#Internal libraries
from aucmedi.ensemble.aggregate import *

#-----------------------------------------------------#
#             Unittest: Aggregate Functions           #
#-----------------------------------------------------#
class AggregateTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create data
        self.pred_data = np.random.rand(10, 8)

    #-------------------------------------------------#
    #           Aggregate: Averaging by Mean          #
    #-------------------------------------------------#
    def test_Aggregate_Mean(self):
        agg_func = AveragingMean()
        self.assertTrue("mean" in aggregate_dict)
        agg_func = aggregate_dict["mean"]()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (8,)))

    #-------------------------------------------------#
    #          Aggregate: Averaging by Median         #
    #-------------------------------------------------#
    def test_Aggregate_Median(self):
        agg_func = AveragingMedian()
        self.assertTrue("median" in aggregate_dict)
        agg_func = aggregate_dict["median"]()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (8,)))

    #-------------------------------------------------#
    #         Aggregate: Majority Vote (Hard)         #
    #-------------------------------------------------#
    def test_Aggregate_MajorityVote(self):
        agg_func = MajorityVote()
        self.assertTrue("majority_vote" in aggregate_dict)
        agg_func = aggregate_dict["majority_vote"]()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (8,)))

    #-------------------------------------------------#
    #                Aggregate: Softmax               #
    #-------------------------------------------------#
    def test_Aggregate_Softmax(self):
        agg_func = Softmax()
        self.assertTrue("softmax" in aggregate_dict)
        agg_func = aggregate_dict["softmax"]()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (8,)))

    #-------------------------------------------------#
    #             Aggregate: Global Argmax            #
    #-------------------------------------------------#
    def test_Aggregate_GlobalArgmax(self):
        agg_func = GlobalArgmax()
        self.assertTrue("global_argmax" in aggregate_dict)
        agg_func = aggregate_dict["global_argmax"]()
        pred = agg_func.aggregate(self.pred_data.copy())
        self.assertTrue(np.array_equal(pred.shape, (8,)))
