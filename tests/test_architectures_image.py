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
from aucmedi.neural_network.architectures.image import *
from aucmedi.neural_network.architectures import supported_standardize_mode as sdm_global
from aucmedi.neural_network.architectures import Classifier, architecture_dict
from aucmedi import *
from aucmedi.data_processing.subfunctions import Resize

#-----------------------------------------------------#
#               Unittest: Architectures               #
#-----------------------------------------------------#
class ArchitecturesImageTEST(unittest.TestCase):
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
        arch = Vanilla(Classifier(n_labels=4), channels=1,
                                    input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Vanilla(Classifier(n_labels=4), channels=3,
                                    input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.Vanilla",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["Vanilla"] == "z-score")
        self.assertTrue(sdm_global["2D.Vanilla"] == "z-score")

    #-------------------------------------------------#
    #              Architecture: ResNet50             #
    #-------------------------------------------------#
    def test_ResNet50(self):
        arch = ResNet50(Classifier(n_labels=4), channels=1,
                                     input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNet50(Classifier(n_labels=4), channels=3,
                                     input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet50",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50"] == "caffe")
        self.assertTrue(sdm_global["2D.ResNet50"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: ResNet101             #
    #-------------------------------------------------#
    def test_ResNet101(self):
        arch = ResNet101(Classifier(n_labels=4), channels=1,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNet101(Classifier(n_labels=4), channels=3,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet101",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101"] == "caffe")
        self.assertTrue(sdm_global["2D.ResNet101"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: ResNet152             #
    #-------------------------------------------------#
    def test_ResNet152(self):
        arch = ResNet152(Classifier(n_labels=4), channels=1,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNet152(Classifier(n_labels=4), channels=3,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet152",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152"] == "caffe")
        self.assertTrue(sdm_global["2D.ResNet152"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: ResNet50V2            #
    #-------------------------------------------------#
    def test_ResNet50V2(self):
        arch = ResNet50V2(Classifier(n_labels=4), channels=1,
                                       input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNet50V2(Classifier(n_labels=4), channels=3,
                                       input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet50V2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50V2"] == "tf")
        self.assertTrue(sdm_global["2D.ResNet50V2"] == "tf")

    #-------------------------------------------------#
    #             Architecture: ResNet101V2           #
    #-------------------------------------------------#
    def test_ResNet101V2(self):
        arch = ResNet101V2(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNet101V2(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet101V2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101V2"] == "tf")
        self.assertTrue(sdm_global["2D.ResNet101V2"] == "tf")

    #-------------------------------------------------#
    #             Architecture: ResNet152V2           #
    #-------------------------------------------------#
    def test_ResNet152V2(self):
        arch = ResNet152V2(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNet152V2(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet152V2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152V2"] == "tf")
        self.assertTrue(sdm_global["2D.ResNet152V2"] == "tf")

    #-------------------------------------------------#
    #             Architecture: ResNeXt50             #
    #-------------------------------------------------#
    def test_ResNeXt50(self):
        arch = ResNeXt50(Classifier(n_labels=4), channels=1,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNeXt50(Classifier(n_labels=4), channels=3,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNeXt50",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt50"] == "torch")
        self.assertTrue(sdm_global["2D.ResNeXt50"] == "torch")

    #-------------------------------------------------#
    #             Architecture: ResNeXt101            #
    #-------------------------------------------------#
    def test_ResNeXt101(self):
        arch = ResNeXt101(Classifier(n_labels=4), channels=1,
                                       input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = ResNeXt101(Classifier(n_labels=4), channels=3,
                                       input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNeXt101",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt101"] == "torch")
        self.assertTrue(sdm_global["2D.ResNeXt101"] == "torch")

    #-------------------------------------------------#
    #            Architecture: DenseNet121            #
    #-------------------------------------------------#
    def test_DenseNet121(self):
        arch = DenseNet121(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = DenseNet121(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.DenseNet121",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet121"] == "torch")
        self.assertTrue(sdm_global["2D.DenseNet121"] == "torch")

    #-------------------------------------------------#
    #            Architecture: DenseNet169            #
    #-------------------------------------------------#
    def test_DenseNet169(self):
        arch = DenseNet169(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = DenseNet169(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.DenseNet169",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet169"] == "torch")
        self.assertTrue(sdm_global["2D.DenseNet169"] == "torch")

    #-------------------------------------------------#
    #            Architecture: DenseNet201            #
    #-------------------------------------------------#
    def test_DenseNet201(self):
        arch = DenseNet201(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = DenseNet201(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.DenseNet201",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet201"] == "torch")
        self.assertTrue(sdm_global["2D.DenseNet201"] == "torch")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB0          #
    #-------------------------------------------------#
    def test_EfficientNetB0(self):
        arch = EfficientNetB0(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB0(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB0",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB0"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB0"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB1          #
    #-------------------------------------------------#
    def test_EfficientNetB1(self):
        arch = EfficientNetB1(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB1(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB1",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB1"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB1"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB2          #
    #-------------------------------------------------#
    def test_EfficientNetB2(self):
        arch = EfficientNetB2(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB2(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB2"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB2"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB3          #
    #-------------------------------------------------#
    def test_EfficientNetB2(self):
        arch = EfficientNetB3(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB3(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB3",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB3"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB3"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB4          #
    #-------------------------------------------------#
    def test_EfficientNetB4(self):
        arch = EfficientNetB4(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB4(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB4",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB4"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB4"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB5          #
    #-------------------------------------------------#
    def test_EfficientNetB5(self):
        arch = EfficientNetB5(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB5(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB5",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB5"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB5"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB6          #
    #-------------------------------------------------#
    def test_EfficientNetB6(self):
        arch = EfficientNetB6(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB6(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB6",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB6"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB6"] == "caffe")

    #-------------------------------------------------#
    #           Architecture: EfficientNetB7          #
    #-------------------------------------------------#
    def test_EfficientNetB7(self):
        arch = EfficientNetB7(Classifier(n_labels=4), channels=1,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = EfficientNetB7(Classifier(n_labels=4), channels=3,
                                           input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB7",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB7"] == "caffe")
        self.assertTrue(sdm_global["2D.EfficientNetB7"] == "caffe")

    #-------------------------------------------------#
    #             Architecture: MobileNet             #
    #-------------------------------------------------#
    def test_MobileNet(self):
        arch = MobileNet(Classifier(n_labels=4), channels=1,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = MobileNet(Classifier(n_labels=4), channels=3,
                                      input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.MobileNet",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNet"] == "tf")
        self.assertTrue(sdm_global["2D.MobileNet"] == "tf")

    #-------------------------------------------------#
    #            Architecture: MobileNetV2            #
    #-------------------------------------------------#
    def test_MobileNetV2(self):
        arch = MobileNetV2(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = MobileNetV2(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.MobileNetV2",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNetV2"] == "tf")
        self.assertTrue(sdm_global["2D.MobileNetV2"] == "tf")

    #-------------------------------------------------#
    #           Architecture: NASNetMobile            #
    #-------------------------------------------------#
    def test_NASNetMobile(self):
        arch = NASNetMobile(Classifier(n_labels=4), channels=1,
                                         input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = NASNetMobile(Classifier(n_labels=4), channels=3,
                                         input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.NASNetMobile",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["NASNetMobile"] == "tf")
        self.assertTrue(sdm_global["2D.NASNetMobile"] == "tf")

    #-------------------------------------------------#
    #            Architecture: NASNetLarge            #
    #-------------------------------------------------#
    def test_NASNetLarge(self):
        arch = NASNetLarge(Classifier(n_labels=4), channels=1,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = NASNetLarge(Classifier(n_labels=4), channels=3,
                                        input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.NASNetLarge",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["NASNetLarge"] == "tf")
        self.assertTrue(sdm_global["2D.NASNetLarge"] == "tf")

    #-------------------------------------------------#
    #         Architecture: InceptionResNetV2         #
    #-------------------------------------------------#
    def test_InceptionResNetV2(self):
        self.datagen_GRAY.sf_resize = Resize(shape=(75, 75))
        self.datagen_RGB.sf_resize = Resize(shape=(75, 75))
        arch = InceptionResNetV2(Classifier(n_labels=4), channels=1,
                                              input_shape=(75, 75))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = InceptionResNetV2(Classifier(n_labels=4), channels=3,
                                              input_shape=(75, 75))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.InceptionResNetV2",
                               batch_queue_size=1, input_shape=(75, 75))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["InceptionResNetV2"] == "tf")
        self.assertTrue(sdm_global["2D.InceptionResNetV2"] == "tf")
        self.datagen_GRAY.sf_resize = Resize(shape=(32, 32))
        self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #            Architecture: InceptionV3            #
    #-------------------------------------------------#
    def test_InceptionV3(self):
        self.datagen_GRAY.sf_resize = Resize(shape=(75, 75))
        self.datagen_RGB.sf_resize = Resize(shape=(75, 75))
        arch = InceptionV3(Classifier(n_labels=4), channels=1,
                                        input_shape=(75, 75))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = InceptionV3(Classifier(n_labels=4), channels=3,
                                        input_shape=(75, 75))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.InceptionV3",
                               batch_queue_size=1, input_shape=(75, 75))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["InceptionV3"] == "tf")
        self.assertTrue(sdm_global["2D.InceptionV3"] == "tf")
        self.datagen_GRAY.sf_resize = Resize(shape=(32, 32))
        self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #               Architecture: VGG16               #
    #-------------------------------------------------#
    def test_VGG16(self):
        arch = VGG16(Classifier(n_labels=4), channels=1,
                                  input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = VGG16(Classifier(n_labels=4), channels=3,
                                  input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.VGG16",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["VGG16"] == "caffe")
        self.assertTrue(sdm_global["2D.VGG16"] == "caffe")

    #-------------------------------------------------#
    #               Architecture: VGG19               #
    #-------------------------------------------------#
    def test_VGG19(self):
        arch = VGG19(Classifier(n_labels=4), channels=1,
                                  input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = VGG19(Classifier(n_labels=4), channels=3,
                                  input_shape=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.VGG19",
                               batch_queue_size=1, input_shape=(32, 32))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["VGG19"] == "caffe")
        self.assertTrue(sdm_global["2D.VGG19"] == "caffe")

    #-------------------------------------------------#
    #              Architecture: Xception             #
    #-------------------------------------------------#
    def test_Xception(self):
        self.datagen_GRAY.sf_resize = Resize(shape=(71, 71))
        self.datagen_RGB.sf_resize = Resize(shape=(71, 71))
        arch = Xception(Classifier(n_labels=4), channels=1,
                                     input_shape=(71, 71))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_GRAY)
        arch = Xception(Classifier(n_labels=4), channels=3,
                                     input_shape=(71, 71))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
                               batch_queue_size=1)
        model.predict(self.datagen_RGB)
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.Xception",
                               batch_queue_size=1, input_shape=(71, 71))
        try : model.model.summary()
        except : raise Exception()
        self.assertTrue(supported_standardize_mode["Xception"] == "tf")
        self.assertTrue(sdm_global["2D.Xception"] == "tf")
        self.datagen_GRAY.sf_resize = Resize(shape=(32, 32))
        self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #              Architecture: ViT B16              #
    #-------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    def test_ViT_B16(self):
        # self.datagen_RGB.sf_resize = Resize(shape=(224, 224))
        # arch = ViT_B16(Classifier(n_labels=4), channels=3,
        #                             input_shape=(224, 224))
        # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
        #                        batch_queue_size=1)
        # model.predict(self.datagen_RGB)
        # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_B16",
        #                        batch_queue_size=1, input_shape=(224, 224))
        # try : model.model.summary()
        # except : raise Exception()
        self.assertTrue(supported_standardize_mode["ViT_B16"] == "tf")
        self.assertTrue(sdm_global["2D.ViT_B16"] == "tf")
        self.assertTrue("2D.ViT_B16" in architecture_dict)
        # self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #              Architecture: ViT B32              #
    #-------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    def test_ViT_B32(self):
        # self.datagen_RGB.sf_resize = Resize(shape=(224, 224))
        # arch = ViT_B32(Classifier(n_labels=4), channels=3,
        #                             input_shape=(224, 224))
        # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
        #                        batch_queue_size=1)
        # model.predict(self.datagen_RGB)
        # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_B32",
        #                        batch_queue_size=1, input_shape=(224, 224))
        # try : model.model.summary()
        # except : raise Exception()
        self.assertTrue(supported_standardize_mode["ViT_B32"] == "tf")
        self.assertTrue(sdm_global["2D.ViT_B32"] == "tf")
        self.assertTrue("2D.ViT_B32" in architecture_dict)
        # self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #              Architecture: ViT L16              #
    #-------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    def test_ViT_L16(self):
        # self.datagen_RGB.sf_resize = Resize(shape=(384, 384))
        # arch = ViT_L16(Classifier(n_labels=4), channels=3,
        #                             input_shape=(384, 384))
        # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
        #                        batch_queue_size=1)
        # model.predict(self.datagen_RGB)
        # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_L16",
        #                        batch_queue_size=1, input_shape=(384, 384))
        # try : model.model.summary()
        # except : raise Exception()
        self.assertTrue(supported_standardize_mode["ViT_L16"] == "tf")
        self.assertTrue(sdm_global["2D.ViT_L16"] == "tf")
        self.assertTrue("2D.ViT_L16" in architecture_dict)
        # self.datagen_RGB.sf_resize = Resize(shape=(32, 32))

    #-------------------------------------------------#
    #              Architecture: ViT L32              #
    #-------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    def test_ViT_L32(self):
        # self.datagen_RGB.sf_resize = Resize(shape=(384, 384))
        # arch = ViT_L32(Classifier(n_labels=4), channels=3,
        #                             input_shape=(384, 384))
        # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch,
        #                        batch_queue_size=1)
        # model.predict(self.datagen_RGB)
        # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_L32",
        #                        batch_queue_size=1, input_shape=(384, 384))
        # try : model.model.summary()
        # except : raise Exception()
        self.assertTrue(supported_standardize_mode["ViT_L32"] == "tf")
        self.assertTrue(sdm_global["2D.ViT_L32"] == "tf")
        self.assertTrue("2D.ViT_L32" in architecture_dict)
        # self.datagen_RGB.sf_resize = Resize(shape=(32, 32))
