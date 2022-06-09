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
import tempfile
import os
from PIL import Image
import numpy as np
#Internal libraries
from aucmedi import DataGenerator, NeuralNetwork, ImageAugmentation, VolumeAugmentation
from aucmedi.data_processing.io_loader import numpy_loader
from aucmedi.ensemble import *

#-----------------------------------------------------#
#                  Unittest: Ensemble                 #
#-----------------------------------------------------#
class EnsembleTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create 2D data
        self.sampleList2D = []
        for i in range(0, 3):
            img = np.random.rand(16, 16, 3) * 255
            img_pillow = Image.fromarray(img.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            path_sample = os.path.join(self.tmp_data.name, index)
            img_pillow.save(path_sample)
            self.sampleList2D.append(index)
        # Create 3D data and DataGenerators
        self.sampleList3D = []
        for i in range(0, 3):
            img_gray = np.random.rand(16, 16, 16) * 255
            index = "image.sample_" + str(i) + ".GRAY.npy"
            path_sampleGRAY = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleGRAY, img_gray)
            self.sampleList3D.append(index)
        # Create classification labels
        self.labels_ohe = np.zeros((3, 2), dtype=np.uint8)
        for i in range(0, 3):
            class_index = np.random.randint(0, 2)
            self.labels_ohe[i][class_index] = 1
        # Initialize model
        self.model2D = NeuralNetwork(n_labels=2, channels=3,
                                      architecture="2D.Vanilla",
                                      batch_queue_size=1,
                                      input_shape=(16, 16))
        self.model3D = NeuralNetwork(n_labels=2, channels=1,
                                      architecture="3D.Vanilla",
                                      batch_queue_size=1,
                                      input_shape=(16, 16, 16))

    #-------------------------------------------------#
    #               Inference Augmenting              #
    #-------------------------------------------------#
    def test_Augmenting_2D_functionality(self):
        # Test functionality with batch_size 10 and n_cycles = 1
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                batch_size=10, resize=None, data_aug=None,
                                grayscale=False, subfunctions=[], standardize_mode="tf")
        preds = predict_augmenting(self.model2D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 2)))

        # Test functionality with batch_size 10 and n_cycles = 5
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                batch_size=10, resize=None, data_aug=None,
                                grayscale=False, subfunctions=[], standardize_mode="tf")
        preds = predict_augmenting(self.model2D, datagen,
                                   n_cycles=5, aggregate="majority_vote")
        self.assertTrue(np.array_equal(preds.shape, (3, 2)))

    def test_Augmenting_2D_customAug(self):
        # Test functionality with batch_size 10 and n_cycles = 1
        my_aug = ImageAugmentation()
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                batch_size=10, resize=None, data_aug=my_aug,
                                grayscale=False, subfunctions=[], standardize_mode="tf")
        preds = predict_augmenting(self.model2D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 2)))

    def test_Augmenting_3D_functionality(self):
        # Test functionality with batch_size 3 and n_cycles = 1
        datagen = DataGenerator(self.sampleList3D, self.tmp_data.name,
                                batch_size=3, resize=None, data_aug=None,
                                grayscale=True, two_dim=False, subfunctions=[],
                                standardize_mode="tf", loader=numpy_loader)
        preds = predict_augmenting(self.model3D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 2)))

        # Test functionality with batch_size 8 and n_cycles = 5
        datagen = DataGenerator(self.sampleList3D, self.tmp_data.name,
                                batch_size=8, resize=None, data_aug=None,
                                grayscale=True, two_dim=False, subfunctions=[],
                                standardize_mode="tf", loader=numpy_loader)
        preds = predict_augmenting(self.model3D, datagen,
                                   n_cycles=5, aggregate="majority_vote")
        self.assertTrue(np.array_equal(preds.shape, (3, 2)))

    def test_Augmenting_3D_customAug(self):
        # Test functionality with self provided augmentation
        my_aug = VolumeAugmentation()
        datagen = DataGenerator(self.sampleList3D, self.tmp_data.name,
                                batch_size=3, resize=None, data_aug=my_aug,
                                grayscale=True, two_dim=False, subfunctions=[],
                                standardize_mode="tf", loader=numpy_loader)
        preds = predict_augmenting(self.model3D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 2)))

    #-------------------------------------------------#
    #                     Bagging                     #
    #-------------------------------------------------#
    def test_Bagging_create(self):
        # Initialize Bagging object
        el = Bagging(model=self.model2D, k_fold=5)
        # Some sanity checks
        self.assertIsInstance(el, Bagging)
        self.assertTrue(el.k_fold == 5)
        self.assertTrue(el.model_template == self.model2D)

    def test_Bagging_training(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                labels=self.labels_ohe, batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Bagging object
        el = Bagging(model=self.model2D, k_fold=3)
        # Run Bagging based training process
        hist = el.train(datagen, epochs=3, iterations=None)

        self.assertIsInstance(hist, dict)
        self.assertTrue("cv_0.loss" in hist and "cv_0.val_loss" in hist)
        self.assertTrue("cv_1.loss" in hist and "cv_1.val_loss" in hist)
        self.assertTrue("cv_2.loss" in hist and "cv_2.val_loss" in hist)

        self.assertTrue(os.path.exists(el.cache_dir.name))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "cv_0.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "cv_0.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "cv_1.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "cv_1.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "cv_2.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "cv_2.model.hdf5")))

        # Delete cached models
        path_tmp_bagging = el.cache_dir.name
        del el
        self.assertFalse(os.path.exists(path_tmp_bagging))

    def test_Bagging_predict(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                labels=self.labels_ohe, batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Bagging object
        el = Bagging(model=self.model2D, k_fold=2)
        # Check cache model directory existence exception
        self.assertRaises(FileNotFoundError, el.predict, datagen)

        # Train model
        el.train(datagen, epochs=1, iterations=None)
        # Run Inference with mean aggregation
        preds = el.predict(datagen, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3,2)))
        # Run Inference with majority vote aggregation
        preds = el.predict(datagen, aggregate="majority_vote")
        self.assertTrue(np.array_equal(preds.shape, (3,2)))
        # Run Inference with returned ensemble
        preds, ensemble = el.predict(datagen, return_ensemble=True)
        self.assertTrue(np.array_equal(preds.shape, (3,2)))
        self.assertTrue(np.array_equal(ensemble.shape, (2,3,2)))

    def test_Bagging_dump(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                labels=self.labels_ohe, batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Bagging object and train it
        el = Bagging(model=self.model2D, k_fold=2)
        el.train(datagen, epochs=1, iterations=None)
        # Initialize temporary directory
        target = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                             suffix=".model")
        self.assertTrue(len(os.listdir(target.name))==0)
        self.assertTrue(len(os.listdir(el.cache_dir.name))==4)
        origin = el.cache_dir.name
        # Dump model
        target_dir = os.path.join(target.name, "test")
        el.dump(target_dir)
        self.assertTrue(len(os.listdir(target_dir))==4)
        self.assertFalse(os.path.exists(origin))
        target_two = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".model")
        target_dir_two = os.path.join(target_two.name, "test")
        el.dump(target_dir_two)
        self.assertTrue(len(os.listdir(target_dir_two))==4)
        self.assertTrue(len(os.listdir(target_dir))==4)
        self.assertTrue(os.path.exists(target_dir))

    def test_Bagging_load(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                labels=self.labels_ohe, batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Bagging object and train it
        el = Bagging(model=self.model2D, k_fold=2)
        el.train(datagen, epochs=1, iterations=None)

        model_dir = el.cache_dir
        el.cache_dir = None
        self.assertRaises(FileNotFoundError, el.predict, datagen)
        self.assertRaises(FileNotFoundError, el.load, "/not/existing/path")
        self.assertRaises(FileNotFoundError, el.load, "/")
        el.load(model_dir.name)
        self.assertTrue(os.path.exists(el.cache_dir))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "cv_0.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "cv_0.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "cv_1.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "cv_1.model.hdf5")))

    #-------------------------------------------------#
    #                    Stacking                     #
    #-------------------------------------------------#
    def test_Stacking_create_metalearner(self):
        # Initialize Stacking object with implemented dictionary
        el = Stacking(model_list=[self.model2D],
                      metalearner="logistic_regression")
        # Some sanity checks
        self.assertIsInstance(el, Stacking)
        from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base
        self.assertIsInstance(el.ml_model, Metalearner_Base)
        # Initialize Stacking object with direct ensembler call
        from aucmedi.ensemble.metalearner import LogisticRegression
        ensembler = LogisticRegression()
        el = Stacking(model_list=[self.model2D],
                      metalearner=ensembler)
        # Some sanity checks
        self.assertIsInstance(el, Stacking)
        from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base
        self.assertIsInstance(el.ml_model, Metalearner_Base)

    def test_Stacking_create_aggregate(self):
        # Initialize Stacking object with implemented dictionary
        el = Stacking(model_list=[self.model2D],
                      metalearner="majority_vote")
        # Some sanity checks
        self.assertIsInstance(el, Stacking)
        from aucmedi.ensemble.aggregate.agg_base import Aggregate_Base
        self.assertIsInstance(el.ml_model, Aggregate_Base)
        # Initialize Stacking object with direct ensembler call
        from aucmedi.ensemble.aggregate import MajorityVote
        ensembler = MajorityVote()
        el = Stacking(model_list=[self.model2D],
                      metalearner=ensembler)
        # Some sanity checks
        self.assertIsInstance(el, Stacking)
        from aucmedi.ensemble.aggregate.agg_base import Aggregate_Base
        self.assertIsInstance(el.ml_model, Aggregate_Base)

    def test_Stacking_create_checks(self):
        el = Stacking(model_list=[self.model2D, self.model2D, self.model2D])
        self.assertTrue(len(el.model_list) == 3)
        self.assertTrue(self.model2D in el.model_list)

    def test_Stacking_training_metalearner(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(np.repeat(self.sampleList2D, 4),
                                self.tmp_data.name,
                                labels=np.repeat(self.labels_ohe, 4, axis=0),
                                batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Stacking object
        el = Stacking(model_list=[self.model2D, self.model2D])
        # Run Stacking based training process
        hist = el.train(datagen, epochs=1, iterations=1)

        self.assertIsInstance(hist, dict)
        self.assertTrue("nn_0.loss" in hist and "nn_0.val_loss" in hist)
        self.assertTrue("nn_1.loss" in hist and "nn_1.val_loss" in hist)

        self.assertTrue(os.path.exists(el.cache_dir.name))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_0.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_0.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_1.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_1.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "metalearner.model.pickle")))
        # Delete cached models
        path_tmp_bagging = el.cache_dir.name
        del el
        self.assertFalse(os.path.exists(path_tmp_bagging))

    def test_Stacking_training_aggregate(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(np.repeat(self.sampleList2D, 4),
                                self.tmp_data.name,
                                labels=np.repeat(self.labels_ohe, 4, axis=0),
                                batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Stacking object
        el = Stacking(model_list=[self.model2D, self.model2D],
                      metalearner="mean")
        # Run Stacking based training process
        hist = el.train(datagen, epochs=1, iterations=1)

        self.assertIsInstance(hist, dict)
        self.assertTrue("nn_0.loss" in hist and "nn_0.val_loss" in hist)
        self.assertTrue("nn_1.loss" in hist and "nn_1.val_loss" in hist)

        self.assertTrue(os.path.exists(el.cache_dir.name))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_0.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_0.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_1.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir.name,
                                                    "nn_1.model.hdf5")))
        # Delete cached models
        path_tmp_bagging = el.cache_dir.name
        del el
        self.assertFalse(os.path.exists(path_tmp_bagging))

    def test_Stacking_predict_metalearner(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(np.repeat(self.sampleList2D, 4),
                                self.tmp_data.name,
                                labels=np.repeat(self.labels_ohe, 4, axis=0),
                                batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Stacking object
        el = Stacking(model_list=[self.model2D, self.model2D])
        # Check cache model directory existence exception
        self.assertRaises(FileNotFoundError, el.predict, datagen)

        # Run Stacking based training process
        hist = el.train(datagen, epochs=1, iterations=1)

        # Run Inference
        preds = el.predict(datagen)
        self.assertTrue(np.array_equal(preds.shape, (12,2)))

        # Run Inference with returned ensemble
        preds, ensemble = el.predict(datagen, return_ensemble=True)
        self.assertTrue(np.array_equal(preds.shape, (12,2)))
        self.assertTrue(np.array_equal(ensemble.shape, (2,12,2)))

    def test_Stacking_predict_aggregate(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(np.repeat(self.sampleList2D, 4),
                                self.tmp_data.name,
                                labels=np.repeat(self.labels_ohe, 4, axis=0),
                                batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Stacking object
        el = Stacking(model_list=[self.model2D, self.model2D],
                      metalearner="mean")
        # Check cache model directory existence exception
        self.assertRaises(FileNotFoundError, el.predict, datagen)

        # Run Stacking based training process
        hist = el.train(datagen, epochs=1, iterations=1)

        # Run Inference
        preds = el.predict(datagen)
        self.assertTrue(np.array_equal(preds.shape, (12,2)))

        # Run Inference with returned ensemble
        preds, ensemble = el.predict(datagen, return_ensemble=True)
        self.assertTrue(np.array_equal(preds.shape, (12,2)))
        self.assertTrue(np.array_equal(ensemble.shape, (2,12,2)))

    def test_Stacking_dump(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(np.repeat(self.sampleList2D, 4),
                                self.tmp_data.name,
                                labels=np.repeat(self.labels_ohe, 4, axis=0),
                                batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Bagging object and train it
        el = Stacking(model_list=[self.model2D])
        el.train(datagen, epochs=1, iterations=1)
        # Initialize temporary directory
        target = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                             suffix=".model")
        self.assertTrue(len(os.listdir(target.name))==0)
        self.assertTrue(len(os.listdir(el.cache_dir.name))==3)
        origin = el.cache_dir.name
        # Dump model
        target_dir = os.path.join(target.name, "test")
        el.dump(target_dir)
        self.assertTrue(len(os.listdir(target_dir))==3)
        self.assertFalse(os.path.exists(origin))
        target_two = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".model")
        target_dir_two = os.path.join(target_two.name, "test")
        el.dump(target_dir_two)
        self.assertTrue(len(os.listdir(target_dir_two))==3)
        self.assertTrue(len(os.listdir(target_dir))==3)
        self.assertTrue(os.path.exists(target_dir))

    def test_Stacking_load(self):
        # Initialize training DataGenerator
        datagen = DataGenerator(np.repeat(self.sampleList2D, 4),
                                self.tmp_data.name,
                                labels=np.repeat(self.labels_ohe, 4, axis=0),
                                batch_size=3, resize=None,
                                data_aug=None, grayscale=False, subfunctions=[],
                                standardize_mode="tf", workers=0)
        # Initialize Bagging object and train it
        el = Stacking(model_list=[self.model2D, self.model2D])
        el.train(datagen, epochs=1, iterations=1)

        model_dir = el.cache_dir
        el.cache_dir = None
        self.assertRaises(FileNotFoundError, el.predict, datagen)
        self.assertRaises(FileNotFoundError, el.load, "/not/existing/path")
        self.assertRaises(FileNotFoundError, el.load, "/")
        el.load(model_dir.name)
        self.assertTrue(os.path.exists(el.cache_dir))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "nn_0.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "nn_0.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "nn_1.logs.csv")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "nn_1.model.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(el.cache_dir,
                                                    "metalearner.model.pickle")))
        preds = el.predict(datagen)
