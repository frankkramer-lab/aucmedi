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
import tempfile
import os
#Internal libraries
from aucmedi.ensemble.metalearner import *

#-----------------------------------------------------#
#                Unittest: Metalearner                #
#-----------------------------------------------------#
class MetelearnerTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create model prediction data
        self.pred_data = np.random.rand(25, 12)
        # Create classification labels
        self.labels_ohe = np.zeros((25, 4), dtype=np.uint8)
        for i in range(0, 25):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1
        # Temporary model directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".model")

    #-------------------------------------------------#
    #               Logistic Regression               #
    #-------------------------------------------------#
    def test_LogisticRegression_create(self):
        # Initializations
        ml = LogisticRegression()
        self.assertTrue("logistic_regression" in metalearner_dict)
        ml = metalearner_dict["logistic_regression"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_LogisticRegression_usage(self):
        # Initializations
        ml = LogisticRegression()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                   Naive Bayes                   #
    #-------------------------------------------------#
    def test_NaiveBayes_create(self):
        # Initializations
        ml = NaiveBayes()
        self.assertTrue("naive_bayes" in metalearner_dict)
        ml = metalearner_dict["naive_bayes"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_NaiveBayes_usage(self):
        # Initializations
        ml = NaiveBayes()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #              Support Vector Machine             #
    #-------------------------------------------------#
    def test_SVM_create(self):
        # Initializations
        ml = SupportVectorMachine()
        self.assertTrue("support_vector_machine" in metalearner_dict)
        ml = metalearner_dict["support_vector_machine"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_SVM_usage(self):
        # Initializations
        ml = SupportVectorMachine()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                 Gaussian Process                #
    #-------------------------------------------------#
    def test_GaussianProcess_create(self):
        # Initializations
        ml = GaussianProcess()
        self.assertTrue("gaussian_process" in metalearner_dict)
        ml = metalearner_dict["gaussian_process"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_GaussianProcess_usage(self):
        # Initializations
        ml = GaussianProcess()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                  Decision Tree                  #
    #-------------------------------------------------#
    def test_DecisionTree_create(self):
        # Initializations
        ml = DecisionTree()
        self.assertTrue("decision_tree" in metalearner_dict)
        ml = metalearner_dict["decision_tree"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_DecisionTree_usage(self):
        # Initializations
        ml = DecisionTree()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                    Best Model                   #
    #-------------------------------------------------#
    def test_BestModel_create(self):
        # Initializations
        ml = BestModel()
        self.assertTrue("best_model" in metalearner_dict)
        ml = metalearner_dict["best_model"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_BestModel_usage(self):
        # Initializations
        ml = BestModel()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                  Weighted Mean                  #
    #-------------------------------------------------#
    def test_AveragingWeightedMean_create(self):
        # Initializations
        ml = AveragingWeightedMean()
        self.assertTrue("weighted_mean" in metalearner_dict)
        ml = metalearner_dict["weighted_mean"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_AveragingWeightedMean_usage(self):
        # Initializations
        ml = AveragingWeightedMean()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                  Random Forest                  #
    #-------------------------------------------------#
    def test_RandomForest_create(self):
        # Initializations
        ml = RandomForest()
        self.assertTrue("random_forest" in metalearner_dict)
        ml = metalearner_dict["random_forest"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_RandomForest_usage(self):
        # Initializations
        ml = RandomForest()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #               k-Nearest Neighbors               #
    #-------------------------------------------------#
    def test_KNearestNeighbors_create(self):
        # Initializations
        ml = KNearestNeighbors()
        self.assertTrue("k_neighbors" in metalearner_dict)
        ml = metalearner_dict["k_neighbors"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_KNearestNeighbors_usage(self):
        # Initializations
        ml = KNearestNeighbors()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))

    #-------------------------------------------------#
    #                MLP Neural Network               #
    #-------------------------------------------------#
    def test_MLP_create(self):
        # Initializations
        ml = MLP()
        self.assertTrue("mlp" in metalearner_dict)
        ml = metalearner_dict["mlp"]()
        # Storage
        model_path = os.path.join(self.tmp_data.name, "ml_model.pickle")
        self.assertFalse(os.path.exists(model_path))
        ml.dump(model_path)
        self.assertTrue(os.path.exists(model_path))
        # Loading
        ml.model = None
        self.assertTrue(ml.model is None)
        ml.load(model_path)
        self.assertFalse(ml.model is None)
        # Cleanup
        os.remove(model_path)

    def test_MLP_usage(self):
        # Initializations
        ml = MLP()
        # Training
        ml.train(x=self.pred_data, y=self.labels_ohe)
        # Inference
        preds = ml.predict(data=self.pred_data)
        # Check
        self.assertTrue(np.array_equal(preds.shape, (25,4)))
