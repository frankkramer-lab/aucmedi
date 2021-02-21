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
from aucmedi.utils.class_weights import *

#-----------------------------------------------------#
#         Unittest: Class Weight Computations         #
#-----------------------------------------------------#
class classweightTEST(unittest.TestCase):
    # Create random classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create classification labels
        self.labels_ohe = np.zeros((25, 4), dtype=np.uint8)
        for i in range(0, 25):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

    #-------------------------------------------------#
    #           Class weights (multi-class)           #
    #-------------------------------------------------#
    def test_classweights_multiclass(self):
        cwl, cwd = compute_class_weights(self.labels_ohe)
        self.assertEqual(len(cwl), 4)
        self.assertTrue(isinstance(cwl[0], float))
        self.assertTrue(isinstance(cwl[1], float))
        self.assertTrue(isinstance(cwl[2], float))
        self.assertTrue(isinstance(cwl[3], float))
        self.assertEqual(len(cwd), 4)
        self.assertTrue(isinstance(cwd[0], float))
        self.assertTrue(isinstance(cwd[1], float))
        self.assertTrue(isinstance(cwd[2], float))
        self.assertTrue(isinstance(cwd[3], float))

    #-------------------------------------------------#
    #           Class weights (multi-label)           #
    #-------------------------------------------------#
    def test_classweights_multilabel(self):
        class_weights = compute_multilabel_weights(self.labels_ohe)
        self.assertEqual(len(class_weights), 4)
        self.assertTrue(isinstance(class_weights[0], float))
        self.assertTrue(isinstance(class_weights[1], float))
        self.assertTrue(isinstance(class_weights[2], float))
        self.assertTrue(isinstance(class_weights[3], float))

    #-------------------------------------------------#
    #                 Sample weights                  #
    #-------------------------------------------------#
    def test_sampleweights(self):
        class_weights = compute_sample_weights(self.labels_ohe)
        self.assertEqual(len(class_weights), 25)
        self.assertTrue(isinstance(class_weights[0], float))
        self.assertTrue(isinstance(class_weights[5], float))
        self.assertTrue(isinstance(class_weights[10], float))
        self.assertTrue(isinstance(class_weights[20], float))
        self.assertTrue(isinstance(class_weights[24], float))
