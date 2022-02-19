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
import numpy as np
#Internal libraries
from aucmedi.neural_network.architectures.volume import *
from aucmedi.neural_network.architectures import supported_standardize_mode as sdm_global
from aucmedi import *
from aucmedi.data_processing.subfunctions import Resize
from aucmedi.data_processing.io_loader import numpy_loader

#-----------------------------------------------------#
#               Unittest: Architectures               #
#-----------------------------------------------------#
class ArchitecturesVolumeTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create HU data
        self.sampleList_hu = []
        for i in range(0, 1):
            img_hu = (np.random.rand(32, 32, 32) * 2000) - 500
            index = "image.sample_" + str(i) + ".HU.npy"
            path_sampleHU = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleHU, img_hu)
            self.sampleList_hu.append(index)

        # Create RGB data
        self.sampleList_rgb = []
        for i in range(0, 1):
            img_rgb = np.random.rand(32, 32, 8, 3) * 255
            index = "image.sample_" + str(i) + ".RGB.npy"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleRGB, img_rgb)
            self.sampleList_rgb.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((1, 4), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create HU Data Generator
        self.datagen_HU = DataGenerator(self.sampleList_hu,
                                        self.tmp_data.name,
                                        labels=self.labels_ohe,
                                        resize=(32, 32, 32),
                                        loader=numpy_loader, two_dim=False,
                                        grayscale=True, batch_size=1)
        # Create RGB Data Generator
        self.datagen_RGB = DataGenerator(self.sampleList_rgb,
                                         self.tmp_data.name,
                                         labels=self.labels_ohe,
                                         resize=(32, 32, 32),
                                         loader=numpy_loader, two_dim=False,
                                         grayscale=False, batch_size=1)

    #-------------------------------------------------#
    #              Architecture: Vanilla              #
    #-------------------------------------------------#
    def test_Vanilla(self):
        arch = Architecture_Vanilla(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_Vanilla(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.Vanilla",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["Vanilla"] == "z-score")
        self.assertTrue(sdm_global["3D.Vanilla"] == "z-score")

    #-------------------------------------------------#
    #            Architecture: DenseNet121            #
    #-------------------------------------------------#
    def test_DenseNet121(self):
        arch = Architecture_DenseNet121(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_DenseNet121(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.DenseNet121",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet121"] == "torch")
        self.assertTrue(sdm_global["3D.DenseNet121"] == "torch")

    #-------------------------------------------------#
    #            Architecture: DenseNet169            #
    #-------------------------------------------------#
    def test_DenseNet169(self):
        arch = Architecture_DenseNet169(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_DenseNet169(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.DenseNet169",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet169"] == "torch")
        self.assertTrue(sdm_global["3D.DenseNet169"] == "torch")

    #-------------------------------------------------#
    #            Architecture: DenseNet201            #
    #-------------------------------------------------#
    def test_DenseNet201(self):
        arch = Architecture_DenseNet201(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_DenseNet201(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.DenseNet201",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet201"] == "torch")
        self.assertTrue(sdm_global["3D.DenseNet201"] == "torch")

    #-------------------------------------------------#
    #              Architecture: ResNet18             #
    #-------------------------------------------------#
    def test_ResNet18(self):
        arch = Architecture_ResNet18(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNet18(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNet18",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet18"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNet18"] == "grayscale")

    #-------------------------------------------------#
    #              Architecture: ResNet34             #
    #-------------------------------------------------#
    def test_ResNet34(self):
        arch = Architecture_ResNet34(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNet34(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNet34",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet34"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNet34"] == "grayscale")

    #-------------------------------------------------#
    #              Architecture: ResNet50             #
    #-------------------------------------------------#
    def test_ResNet50(self):
        arch = Architecture_ResNet50(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNet50(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNet50",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNet50"] == "grayscale")

    #-------------------------------------------------#
    #             Architecture: ResNet101             #
    #-------------------------------------------------#
    def test_ResNet101(self):
        arch = Architecture_ResNet101(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNet101(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNet101",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNet101"] == "grayscale")

    #-------------------------------------------------#
    #             Architecture: ResNet152             #
    #-------------------------------------------------#
    def test_ResNet152(self):
        arch = Architecture_ResNet152(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNet152(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNet152",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNet152"] == "grayscale")

    #-------------------------------------------------#
    #             Architecture: ResNeXt50             #
    #-------------------------------------------------#
    def test_ResNeXt50(self):
        arch = Architecture_ResNeXt50(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNeXt50(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNeXt50",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt50"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNeXt50"] == "grayscale")

    #-------------------------------------------------#
    #            Architecture: ResNeXt101             #
    #-------------------------------------------------#
    def test_ResNeXt101(self):
        arch = Architecture_ResNeXt101(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_ResNeXt101(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.ResNeXt101",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt101"] == "grayscale")
        self.assertTrue(sdm_global["3D.ResNeXt101"] == "grayscale")

    #-------------------------------------------------#
    #               Architecture: VGG16               #
    #-------------------------------------------------#
    def test_VGG16(self):
        arch = Architecture_VGG16(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_VGG16(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.VGG16",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["VGG16"] == "caffe")
        self.assertTrue(sdm_global["3D.VGG16"] == "caffe")

    #-------------------------------------------------#
    #               Architecture: VGG19               #
    #-------------------------------------------------#
    def test_VGG19(self):
        arch = Architecture_VGG19(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_VGG19(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.VGG19",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["VGG19"] == "caffe")
        self.assertTrue(sdm_global["3D.VGG19"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: MobileNet             #
    #-------------------------------------------------#
    def test_MobileNet(self):
        arch = Architecture_MobileNet(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_MobileNet(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.MobileNet",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNet"] == "tf")
        self.assertTrue(sdm_global["3D.MobileNet"] == "tf")

    #-------------------------------------------------#
    #            Architecture: MobileNetV2            #
    #-------------------------------------------------#
    def test_MobileNetV2(self):
        arch = Architecture_MobileNetV2(channels=1, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_HU)
        arch = Architecture_MobileNetV2(channels=3, input_shape=(32, 32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="3D.MobileNetV2",
                               batch_queue_size=1, input_shape=(32, 32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNetV2"] == "tf")
        self.assertTrue(sdm_global["3D.MobileNetV2"] == "tf")
