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
import tempfile
import os
from PIL import Image
import numpy as np
#Internal libraries
from aucmedi.neural_network.architectures import *
from aucmedi import *

#-----------------------------------------------------#
#               Unittest: Architectures               #
#-----------------------------------------------------#
class ArchitecturesTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create Grayscale data
        self.sampleList_gray = []
        for i in range(0, 1):
            img_gray = np.random.rand(32, 32) * 255
            imgGRAY_pillow = Image.fromarray(img_gray.astype(np.uint8))
            index = "image.sample_" + str(i) + ".GRAY.png"
            path_sampleGRAY = os.path.join(self.tmp_data.name, index)
            imgGRAY_pillow.save(path_sampleGRAY)
            self.sampleList_gray.append(index)

        # Create RGB data
        self.sampleList_rgb = []
        for i in range(0, 1):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((1, 4), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create Grayscale Data Generator
        self.datagen_GRAY = DataGenerator(self.sampleList_gray,
                                          self.tmp_data.name,
                                          labels=self.labels_ohe,
                                          grayscale=True, batch_size=1)
        # Create RGB Data Generator
        self.datagen_RGB = DataGenerator(self.sampleList_rgb,
                                         self.tmp_data.name,
                                         labels=self.labels_ohe,
                                         grayscale=False, batch_size=1)

    #-------------------------------------------------#
    #              Architecture: Vanilla              #
    #-------------------------------------------------#
    def test_Vanilla(self):
        arch = Architecture_Vanilla(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_Vanilla(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="Vanilla",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["Vanilla"] == "tf")

    #-------------------------------------------------#
    #              Architecture: ResNet50             #
    #-------------------------------------------------#
    def test_ResNet50(self):
        arch = Architecture_ResNet50(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNet50(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNet50",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50"] == "caffe")

    #-------------------------------------------------#
    #            Architecture: DenseNet121            #
    #-------------------------------------------------#
    def test_DenseNet121(self):
        arch = Architecture_DenseNet121(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_DenseNet121(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="DenseNet121",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet121"] == "torch")

    #-------------------------------------------------#
    #            Architecture: DenseNet169            #
    #-------------------------------------------------#
    def test_DenseNet169(self):
        arch = Architecture_DenseNet169(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_DenseNet169(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="DenseNet169",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet169"] == "torch")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB2          #
    #-------------------------------------------------#
    def test_EfficientNetB2(self):
        arch = Architecture_EfficientNetB2(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB2(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB2"] == "caffe")
