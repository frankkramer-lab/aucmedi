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
from aucmedi import *

#-----------------------------------------------------#
#              Unittest: NeuralNetwork               #
#-----------------------------------------------------#
class NeuralNetworkTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")

        # Create RGB data
        self.sampleList_rgb = []
        for i in range(0, 3):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((3, 4), dtype=np.uint8)
        for i in range(0, 3):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create RGB Data Generator
        self.datagen = DataGenerator(self.sampleList_rgb,
                                     self.tmp_data.name,
                                     labels=self.labels_ohe,
                                     resize=(32, 32),
                                     grayscale=False)

    #-------------------------------------------------#
    #                  Model Training                 #
    #-------------------------------------------------#
    def test_training_pure(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        hist = model.train(training_generator=self.datagen, 
                           epochs=3, batch_size=2)
        self.assertTrue("loss" in hist)
        self.assertTrue(len(hist["loss"]) == 3)

    def test_training_iterations(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        hist = model.train(training_generator=self.datagen,
                           epochs=3, iterations=5)
        self.assertTrue("loss" in hist)
        self.assertTrue(len(hist["loss"]) == 3)

    def test_training_validation(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        hist = model.train(training_generator=self.datagen,
                           validation_generator=self.datagen,
                           epochs=3, batch_size=2)
        self.assertTrue("loss" in hist and "val_loss" in hist)
        self.assertTrue(len(hist["loss"]) == 3)

    def test_training_transferlearning(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        model.tf_epochs = 2
        hist = model.train(training_generator=self.datagen,
                           validation_generator=self.datagen,
                           epochs=3, batch_size=2, transfer_learning=True)
        self.assertTrue("tl_loss" in hist and "tl_val_loss" in hist)
        self.assertTrue(len(hist["tl_loss"])==2)
        self.assertTrue("ft_loss" in hist and "ft_val_loss" in hist)
        self.assertTrue(len(hist["ft_loss"])==1)

    def test_training_metadata(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32),
                              meta_variables=5)
        datagen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                labels=self.labels_ohe, resize=(32, 32), 
                                grayscale=False,
                                metadata=np.zeros((3, 5), dtype=np.float32))
        hist = model.train(training_generator=datagen,
                           validation_generator=datagen,
                           epochs=3, batch_size=2)
        self.assertTrue("loss" in hist and "val_loss" in hist)
        self.assertTrue(len(hist["loss"]) == 3)

    #-------------------------------------------------#
    #                 Model Inference                 #
    #-------------------------------------------------#
    def test_predict(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        preds = model.predict(self.datagen, batch_size=1)
        self.assertTrue(preds.shape == (3, 4))
        self.assertTrue(np.sum(preds[0]) >= 0.99 and np.sum(preds[0]) <= 1.01)

    def test_predict_metadata(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32),
                              meta_variables=5)
        datagen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                labels=None, resize=(32, 32), 
                                grayscale=False,
                                metadata=np.zeros((3, 5), dtype=np.float32))
        preds = model.predict(datagen, batch_size=2)
        self.assertTrue(preds.shape == (3, 4))
        self.assertTrue(np.sum(preds[0]) >= 0.99 and np.sum(preds[0]) <= 1.01)

    #-------------------------------------------------#
    #             Dataset Transformation              #
    #-------------------------------------------------#
    def test_dataset_build(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        ds = model.__prepare_dataset__(self.datagen, batch_size=2)

    def test_dataset_training(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        ds = model.__prepare_dataset__(self.datagen, batch_size=2)
        it = iter(ds)
        batch_first = it.get_next()
        self.assertTrue(len(batch_first) == 2)
        self.assertTrue(np.array_equal(batch_first[0].shape, (2,32,32,3)))
        self.assertTrue(np.array_equal(batch_first[1].shape, (2,4)))
        batch_second = it.get_next()
        self.assertTrue(len(batch_second) == 2)
        self.assertTrue(np.array_equal(batch_second[0].shape, (1,32,32,3)))
        self.assertTrue(np.array_equal(batch_second[1].shape, (1,4)))

    def test_dataset_prediction(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32))
        datagen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                labels=None, resize=(32, 32), grayscale=False)
        ds = model.__prepare_dataset__(datagen, batch_size=2)
        it = iter(ds)
        batch_first = it.get_next()
        self.assertTrue(len(batch_first) == 1)
        self.assertTrue(np.array_equal(batch_first[0].shape, (2,32,32,3)))
        batch_second = it.get_next()
        self.assertTrue(len(batch_second) == 1)
        self.assertTrue(np.array_equal(batch_second[0].shape, (1,32,32,3)))

    def test_dataset_training_meta(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32),
                              meta_variables=5)
        datagen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                labels=self.labels_ohe, resize=(32, 32), 
                                grayscale=False,
                                metadata=np.zeros((3, 5), dtype=np.float32))
        ds = model.__prepare_dataset__(datagen, batch_size=2)
        it = iter(ds)
        batch_first = it.get_next()
        self.assertTrue(len(batch_first) == 2)
        self.assertTrue(np.array_equal(batch_first[0][0].shape, (2,32,32,3)))
        self.assertTrue(np.array_equal(batch_first[0][1].shape, (2,5)))
        self.assertTrue(np.array_equal(batch_first[1].shape, (2,4)))
        batch_second = it.get_next()
        self.assertTrue(len(batch_second) == 2)
        self.assertTrue(np.array_equal(batch_second[0][0].shape, (1,32,32,3)))
        self.assertTrue(np.array_equal(batch_second[0][1].shape, (1,5)))
        self.assertTrue(np.array_equal(batch_second[1].shape, (1,4)))

    def test_dataset_prediction_meta(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32),
                              meta_variables=5)
        datagen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                labels=None, resize=(32, 32), 
                                grayscale=False,
                                metadata=np.zeros((3, 5), dtype=np.float32))
        ds = model.__prepare_dataset__(datagen, batch_size=2)
        it = iter(ds)
        batch_first = it.get_next()
        self.assertTrue(len(batch_first) == 1)
        self.assertTrue(np.array_equal(batch_first[0][0].shape, (2,32,32,3)))
        self.assertTrue(np.array_equal(batch_first[0][1].shape, (2,5)))
        batch_second = it.get_next()
        self.assertTrue(len(batch_second) == 1)
        self.assertTrue(np.array_equal(batch_second[0][0].shape, (1,32,32,3)))
        self.assertTrue(np.array_equal(batch_second[0][1].shape, (1,5)))

    def test_dataset_training_weights(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32),
                              meta_variables=5)
        datagen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                labels=self.labels_ohe, resize=(32, 32), 
                                grayscale=False,
                                sample_weights=np.random.random((3,)))
        ds = model.__prepare_dataset__(datagen, batch_size=2)
        it = iter(ds)
        batch_first = it.get_next()
        self.assertTrue(len(batch_first) == 3)
        self.assertTrue(np.array_equal(batch_first[0].shape, (2,32,32,3)))
        self.assertTrue(np.array_equal(batch_first[1].shape, (2,4)))
        self.assertTrue(np.array_equal(batch_first[2].shape, (2,)))
        batch_second = it.get_next()
        self.assertTrue(len(batch_second) == 3)
        self.assertTrue(np.array_equal(batch_second[0].shape, (1,32,32,3)))
        self.assertTrue(np.array_equal(batch_second[1].shape, (1,4)))
        self.assertTrue(np.array_equal(batch_second[2].shape, (1,)))
