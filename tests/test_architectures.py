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
from aucmedi.data_processing.subfunctions import Resize

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
                                          resize=(32, 32),
                                          grayscale=True, batch_size=1)
        # Create RGB Data Generator
        self.datagen_RGB = DataGenerator(self.sampleList_rgb,
                                         self.tmp_data.name,
                                         labels=self.labels_ohe,
                                         resize=(32, 32),
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
    #             Architecture: ResNet101             #
    #-------------------------------------------------#
    def test_ResNet101(self):
        arch = Architecture_ResNet101(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNet101(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNet101",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: ResNet152             #
    #-------------------------------------------------#
    def test_ResNet152(self):
        arch = Architecture_ResNet152(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNet152(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNet152",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: ResNet50V2            #
    #-------------------------------------------------#
    def test_ResNet50V2(self):
        arch = Architecture_ResNet50V2(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNet50V2(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNet50V2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50V2"] == "tf")

    #-------------------------------------------------#
    #             Architecture: ResNet101V2           #
    #-------------------------------------------------#
    def test_ResNet101V2(self):
        arch = Architecture_ResNet101V2(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNet101V2(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNet101V2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101V2"] == "tf")

    #-------------------------------------------------#
    #             Architecture: ResNet152V2           #
    #-------------------------------------------------#
    def test_ResNet152V2(self):
        arch = Architecture_ResNet152V2(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNet152V2(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNet152V2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152V2"] == "tf")

    #-------------------------------------------------#
    #             Architecture: ResNeXt50             #
    #-------------------------------------------------#
    def test_ResNeXt50(self):
        arch = Architecture_ResNeXt50(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNeXt50(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNeXt50",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt50"] == "torch")

    #-------------------------------------------------#
    #             Architecture: ResNeXt101            #
    #-------------------------------------------------#
    def test_ResNeXt101(self):
        arch = Architecture_ResNeXt101(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_ResNeXt101(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="ResNeXt101",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt101"] == "torch")

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
    #            Architecture: DenseNet201            #
    #-------------------------------------------------#
    def test_DenseNet201(self):
        arch = Architecture_DenseNet201(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_DenseNet201(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="DenseNet201",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet201"] == "torch")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB0          #
    #-------------------------------------------------#
    def test_EfficientNetB0(self):
        arch = Architecture_EfficientNetB0(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB0(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB0",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB0"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB1          #
    #-------------------------------------------------#
    def test_EfficientNetB1(self):
        arch = Architecture_EfficientNetB1(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB1(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB1",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB1"] == "caffe")

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

    #-------------------------------------------------#
    #           Architecture: EfficientNetB3          #
    #-------------------------------------------------#
    def test_EfficientNetB2(self):
        arch = Architecture_EfficientNetB3(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB3(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB3",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB3"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB4          #
    #-------------------------------------------------#
    def test_EfficientNetB4(self):
        arch = Architecture_EfficientNetB4(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB4(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB4",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB4"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB5          #
    #-------------------------------------------------#
    def test_EfficientNetB5(self):
        arch = Architecture_EfficientNetB5(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB5(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB5",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB5"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB6          #
    #-------------------------------------------------#
    def test_EfficientNetB6(self):
        arch = Architecture_EfficientNetB6(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB6(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB6",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB6"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB7          #
    #-------------------------------------------------#
    def test_EfficientNetB7(self):
        arch = Architecture_EfficientNetB7(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_EfficientNetB7(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="EfficientNetB7",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB7"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: MobileNet             #
    #-------------------------------------------------#
    def test_MobileNet(self):
        arch = Architecture_MobileNet(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_MobileNet(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="MobileNet",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNet"] == "tf")

    #-------------------------------------------------#
    #            Architecture: MobileNetV2            #
    #-------------------------------------------------#
    def test_MobileNetV2(self):
        arch = Architecture_MobileNetV2(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_MobileNetV2(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="MobileNetV2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNetV2"] == "tf")

    #-------------------------------------------------#
    #           Architecture: NASNetMobile            #
    #-------------------------------------------------#
    def test_NASNetMobile(self):
        arch = Architecture_NASNetMobile(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_NASNetMobile(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="NASNetMobile",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["NASNetMobile"] == "tf")

    #-------------------------------------------------#
    #            Architecture: NASNetLarge            #
    #-------------------------------------------------#
    def test_NASNetLarge(self):
        arch = Architecture_NASNetLarge(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_NASNetLarge(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="NASNetLarge",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["NASNetLarge"] == "tf")

    #-------------------------------------------------#
    #         Architecture: InceptionResNetV2         #
    #-------------------------------------------------#
    def test_InceptionResNetV2(self):
        self.datagen_GRAY.sf_resize = Resize(shape=(75, 75))
        self.datagen_RGB.sf_resize = Resize(shape=(75, 75))
        arch = Architecture_InceptionResNetV2(channels=1, input_shape=(75, 75))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_InceptionResNetV2(channels=3, input_shape=(75, 75))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="InceptionResNetV2",
                               batch_queue_size=1, input_shape=(75, 75))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["InceptionResNetV2"] == "tf")
        self.datagen_GRAY.sf_resize = Resize(shape=(32, 32))
        self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #            Architecture: InceptionV3            #
    #-------------------------------------------------#
    def test_InceptionV3(self):
        self.datagen_GRAY.sf_resize = Resize(shape=(75, 75))
        self.datagen_RGB.sf_resize = Resize(shape=(75, 75))
        arch = Architecture_InceptionV3(channels=1, input_shape=(75, 75))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_InceptionV3(channels=3, input_shape=(75, 75))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="InceptionV3",
                               batch_queue_size=1, input_shape=(75, 75))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["InceptionV3"] == "tf")
        self.datagen_GRAY.sf_resize = Resize(shape=(32, 32))
        self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #               Architecture: VGG16               #
    #-------------------------------------------------#
    def test_VGG16(self):
        arch = Architecture_VGG16(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_VGG16(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="VGG16",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["VGG16"] == "caffe")

    #-------------------------------------------------#
    #               Architecture: VGG19               #
    #-------------------------------------------------#
    def test_VGG19(self):
        arch = Architecture_VGG19(channels=1, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_VGG19(channels=3, input_shape=(32, 32))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="VGG19",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["VGG19"] == "caffe")

    #-------------------------------------------------#
    #              Architecture: Xception             #
    #-------------------------------------------------#
    def test_Xception(self):
        self.datagen_GRAY.sf_resize = Resize(shape=(71, 71))
        self.datagen_RGB.sf_resize = Resize(shape=(71, 71))
        arch = Architecture_Xception(channels=1, input_shape=(71, 71))
        model = Neural_Network(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Architecture_Xception(channels=3, input_shape=(71, 71))
        model = Neural_Network(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = Neural_Network(n_labels=4, channels=3, architecture="Xception",
                               batch_queue_size=1, input_shape=(71, 71))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["Xception"] == "tf")
        self.datagen_GRAY.sf_resize = Resize(shape=(32, 32))
        self.datagen_RGB.sf_resize = Resize(shape=(32, 32))
