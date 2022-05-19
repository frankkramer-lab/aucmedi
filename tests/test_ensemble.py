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
from aucmedi import DataGenerator, Neural_Network, Image_Augmentation, Volume_Augmentation
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
        self.labels_ohe = np.zeros((3, 4), dtype=np.uint8)
        for i in range(0, 3):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1
        # Initialize model
        self.model2D = Neural_Network(n_labels=4, channels=3,
                                      architecture="2D.Vanilla",
                                      batch_queue_size=1,
                                      input_shape=(16, 16))
        self.model3D = Neural_Network(n_labels=4, channels=1,
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
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

        # Test functionality with batch_size 10 and n_cycles = 5
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                batch_size=10, resize=None, data_aug=None,
                                grayscale=False, subfunctions=[], standardize_mode="tf")
        preds = predict_augmenting(self.model2D, datagen,
                                   n_cycles=5, aggregate="majority_vote")
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

    def test_Augmenting_2D_customAug(self):
        # Test functionality with batch_size 10 and n_cycles = 1
        my_aug = Image_Augmentation()
        datagen = DataGenerator(self.sampleList2D, self.tmp_data.name,
                                batch_size=10, resize=None, data_aug=my_aug,
                                grayscale=False, subfunctions=[], standardize_mode="tf")
        preds = predict_augmenting(self.model2D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

    def test_Augmenting_3D_functionality(self):
        # Test functionality with batch_size 3 and n_cycles = 1
        datagen = DataGenerator(self.sampleList3D, self.tmp_data.name,
                                batch_size=3, resize=None, data_aug=None,
                                grayscale=True, two_dim=False, subfunctions=[],
                                standardize_mode="tf", loader=numpy_loader)
        preds = predict_augmenting(self.model3D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

        # Test functionality with batch_size 8 and n_cycles = 5
        datagen = DataGenerator(self.sampleList3D, self.tmp_data.name,
                                batch_size=8, resize=None, data_aug=None,
                                grayscale=True, two_dim=False, subfunctions=[],
                                standardize_mode="tf", loader=numpy_loader)
        preds = predict_augmenting(self.model3D, datagen,
                                   n_cycles=5, aggregate="majority_vote")
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

    def test_Augmenting_3D_customAug(self):
        # Test functionality with self provided augmentation
        my_aug = Volume_Augmentation()
        datagen = DataGenerator(self.sampleList3D, self.tmp_data.name,
                                batch_size=3, resize=None, data_aug=my_aug,
                                grayscale=True, two_dim=False, subfunctions=[],
                                standardize_mode="tf", loader=numpy_loader)
        preds = predict_augmenting(self.model3D, datagen,
                                   n_cycles=1, aggregate="mean")
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

    #-------------------------------------------------#
    #                     Bagging                     #
    #-------------------------------------------------#
    def test_Bagging_create(self):
        # Initialize Bagging object
        el = Bagging(model=self.model2D, k_fold=5)
        # Some sanity checks
        self.assertIsInstance(el, Bagging)
        self.assertTrue(len(el.model_list) == 5)
        self.assertTrue(el.model_list[2] != self.model2D)
        self.assertTrue(el.model_list[1].n_labels == self.model2D.n_labels)
        self.assertTrue(el.model_list[0].meta_input == self.model2D.meta_input)

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
        self.assertTrue(np.array_equal(preds.shape, (3,4)))
        # Run Inference with majority vote aggregation
        preds = el.predict(datagen, aggregate="majority_vote")
        self.assertTrue(np.array_equal(preds.shape, (3,4)))

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
