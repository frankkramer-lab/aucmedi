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
from aucmedi.neural_network.architectures import Classifier
from aucmedi.neural_network.architectures.image import Vanilla
from tensorflow.keras import Input

#-----------------------------------------------------#
#            Unittest: Classification Head            #
#-----------------------------------------------------#
class ClassifierTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")

        # Create data
        self.sampleList = []
        for i in range(0, 1):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((1, 20), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 20)
            self.labels_ohe[i][class_index] = 1

        # Create metadata
        self.metadata = np.zeros((1, 10), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 10)
            self.metadata[i][class_index] = 1

        # Create Data Generator
        self.datagen = DataGenerator(self.sampleList,
                                     self.tmp_data.name,
                                     labels=self.labels_ohe,
                                     resize=(32, 32),
                                     grayscale=False, batch_size=1)

        # Create Data Generator with Metadata
        self.datagen_meta = DataGenerator(self.sampleList,
                                          self.tmp_data.name,
                                          labels=self.labels_ohe,
                                          metadata=self.metadata,
                                          resize=(32, 32),
                                          grayscale=False, batch_size=1)

    #-------------------------------------------------#
    #           Initialization Functionality          #
    #-------------------------------------------------#
    def test_create(self):
        classification_head = Classifier(n_labels=20, fcl_dropout=True,
                                         activation_output="softmax")
        self.assertIsInstance(classification_head, Classifier)

    #-------------------------------------------------#
    #            Application - Multi-Class            #
    #-------------------------------------------------#
    def test_application_multiclass(self):
        model = NeuralNetwork(n_labels=20, channels=3, batch_queue_size=1,
                               input_shape=(32, 32), activation_output="softmax")
        preds = model.predict(self.datagen)
        self.assertTrue(np.sum(preds[0]) > 0.99 and np.sum(preds[0]) < 1.01)

    #-------------------------------------------------#
    #            Application - Multi-Label            #
    #-------------------------------------------------#
    def test_application_multilabel(self):
        model = NeuralNetwork(n_labels=20, channels=3, batch_queue_size=1,
                               input_shape=(32, 32), activation_output="sigmoid")
        preds = model.predict(self.datagen)
        self.assertTrue(np.sum(preds[0]) > 5)

    #-------------------------------------------------#
    #              Application - Metadata             #
    #-------------------------------------------------#
    def test_application_multilabel(self):
        model = NeuralNetwork(n_labels=20, channels=3, batch_queue_size=1,
                               input_shape=(32, 32), activation_output="softmax",
                               meta_variables=10)
        preds = model.predict(self.datagen_meta)
        self.assertTrue(np.sum(preds[0]) > 0.99 and np.sum(preds[0]) < 1.01)

    #-------------------------------------------------#
    #          Architecture Interoperability          #
    #-------------------------------------------------#
    def test_interoperability(self):
        classification_head = Classifier(n_labels=20, fcl_dropout=True,
                                         activation_output="softmax")
        arch = Vanilla(classification_head, channels=3,
                                    input_shape=(32, 32))
        model = arch.create_model()
        try : model.summary()
        except : raise Exception()

    #-------------------------------------------------#
    #                  Functionality                  #
    #-------------------------------------------------#
    def test_functionality_base(self):
        classification_head = Classifier(n_labels=20, fcl_dropout=True,
                                         activation_output="softmax")
        model_input = Input(shape=(32,32,3))
        model = classification_head.build(model_input=model_input,
                                          model_output=model_input)
        try : model.summary()
        except : raise Exception()

    #-------------------------------------------------#
    #             Functionality - Metadata            #
    #-------------------------------------------------#
    def test_functionality_metadata(self):
        classification_head = Classifier(n_labels=20, fcl_dropout=True,
                                         activation_output="softmax",
                                         meta_variables=10)
        model_input = Input(shape=(32,32,3))
        model = classification_head.build(model_input=model_input,
                                          model_output=model_input)
        try : model.summary()
        except : raise Exception()
