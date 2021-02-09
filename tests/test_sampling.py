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
from sklearn.datasets import make_classification
#Internal libraries
from aucmedi.sampling import sampling_split, sampling_kfold

#-----------------------------------------------------#
#                  Unittest: Sampling                 #
#-----------------------------------------------------#
class SamplingTEST(unittest.TestCase):
    # Create random classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Generate data
        self.x, y = make_classification(n_samples=1000, n_features=4,
                                        random_state=0, shuffle=False)
        # One-hot encode labels
        self.y = np.zeros((y.size, 2))
        self.y[np.arange(y.size), y] = 1

    #-------------------------------------------------#
    #          Sampling via Percentage Split          #
    #-------------------------------------------------#
    # Check percentage split ratio exception handling
    def test_PercentageSplit_samplingCheck(self):
        self.assertRaises(ValueError, sampling_split, self.x, self.y,
                          sampling=[0.8, 0.05, 0.1])

    # Check random sampling via percentage split
    def test_PercentageSplit_random(self):
        subsets = sampling_split(self.x, self.y,
                                 sampling=[0.4, 0.3, 0.05, 0.25],
                                 iterative=False, stratified=False)
        self.assertTrue(subsets[0][0].shape[0] > 395 and \
                        subsets[0][0].shape[0] < 405)
        self.assertTrue(subsets[0][1].shape[0] > 395 and \
                        subsets[0][1].shape[0] < 405)
        self.assertTrue(subsets[1][0].shape[0] > 295 and \
                        subsets[1][0].shape[0] < 305)
        self.assertTrue(subsets[2][0].shape[0] > 45 and \
                        subsets[2][0].shape[0] < 55)
        self.assertTrue(subsets[3][0].shape[0] > 245 and \
                        subsets[3][0].shape[0] < 255)

    # Check stratified random sampling via percentage split
    def test_PercentageSplit_stratified(self):
      subsets = sampling_split(self.x, self.y,
                               sampling=[0.4, 0.3, 0.05, 0.25],
                               iterative=False, stratified=True)
      self.assertTrue(subsets[0][0].shape[0] > 395 and \
                      subsets[0][0].shape[0] < 405)
      self.assertTrue(subsets[0][1].shape[0] > 395 and \
                      subsets[0][1].shape[0] < 405)
      self.assertTrue(subsets[1][0].shape[0] > 295 and \
                      subsets[1][0].shape[0] < 305)
      self.assertTrue(subsets[2][0].shape[0] > 45 and \
                      subsets[2][0].shape[0] < 55)
      self.assertTrue(subsets[3][0].shape[0] > 245 and \
                      subsets[3][0].shape[0] < 255)

    # Check stratified iterative sampling via percentage split
    def test_PercentageSplit_iterative(self):
        subsets = sampling_split(self.x, self.y,
                                 sampling=[0.4, 0.3, 0.05, 0.25],
                                 iterative=True, stratified=True)
        self.assertTrue(subsets[0][0].shape[0] > 395 and \
                        subsets[0][0].shape[0] < 405)
        self.assertTrue(subsets[0][1].shape[0] > 395 and \
                        subsets[0][1].shape[0] < 405)
        self.assertTrue(subsets[1][0].shape[0] > 295 and \
                        subsets[1][0].shape[0] < 305)
        self.assertTrue(subsets[2][0].shape[0] > 45 and \
                        subsets[2][0].shape[0] < 55)
        self.assertTrue(subsets[3][0].shape[0] > 245 and \
                        subsets[3][0].shape[0] < 255)

    #-------------------------------------------------#
    #       Sampling via k-fold Cross-Validation      #
    #-------------------------------------------------#
    # Check random sampling via k-fold cross-validation
    def test_CrossValidation_random(self):
        subsets = sampling_kfold(self.x, self.y, n_splits=5,
                                 iterative=False, stratified=False)
        self.assertEqual(len(subsets), 5)
        for fold in subsets:
            (tx, ty, vx, vy) = fold
            self.assertTrue(tx.shape[0] > 795 and tx.shape[0] < 805)
            self.assertTrue(ty.shape[0] > 795 and ty.shape[0] < 805)
            self.assertTrue(vx.shape[0] > 195 and vx.shape[0] < 205)
            self.assertTrue(vy.shape[0] > 195 and vy.shape[0] < 205)

    # Check stratified random sampling via k-fold cross-validation
    def test_CrossValidation_stratified(self):
        subsets = sampling_kfold(self.x, self.y, n_splits=5,
                                 iterative=False, stratified=True)
        self.assertEqual(len(subsets), 5)
        for fold in subsets:
            (tx, ty, vx, vy) = fold
            self.assertTrue(tx.shape[0] > 795 and tx.shape[0] < 805)
            self.assertTrue(ty.shape[0] > 795 and ty.shape[0] < 805)
            self.assertTrue(vx.shape[0] > 195 and vx.shape[0] < 205)
            self.assertTrue(vy.shape[0] > 195 and vy.shape[0] < 205)

    # Check stratified iterative sampling via k-fold cross-validation
    def test_CrossValidation_iterative(self):
        subsets = sampling_kfold(self.x, self.y, n_splits=5,
                                 iterative=True, stratified=True)
        self.assertEqual(len(subsets), 5)
        for fold in subsets:
            (tx, ty, vx, vy) = fold
            self.assertTrue(tx.shape[0] > 795 and tx.shape[0] < 805)
            self.assertTrue(ty.shape[0] > 795 and ty.shape[0] < 805)
            self.assertTrue(vx.shape[0] > 195 and vx.shape[0] < 205)
            self.assertTrue(vy.shape[0] > 195 and vy.shape[0] < 205)
