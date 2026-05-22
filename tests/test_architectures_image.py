# ==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import unittest
import tempfile
import os
from PIL import Image
import numpy as np

# Internal libraries
from aucmedi import NeuralNetwork
from aucmedi.neural_network.architectures.image import *
from aucmedi.neural_network.architectures import (
    supported_standardize_mode as sdm_global,
)
from aucmedi.neural_network.architectures import Classifier, architecture_dict
from aucmedi import *
from aucmedi.data_processing.subfunctions import Resize


# -----------------------------------------------------#
#               Unittest: Architectures               #
# -----------------------------------------------------#
class ArchitecturesImageTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )
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
        self.dataloader_GRAY = create_batch_loader(
            self.sampleList_gray,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(32, 32),
            grayscale=True,
            batch_size=1,
        )
        # Create RGB Data Generator
        self.dataloader_RGB = create_batch_loader(
            self.sampleList_rgb,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(32, 32),
            grayscale=False,
            batch_size=1,
        )

    # -------------------------------------------------#
    #              Architecture: Vanilla              #
    # -------------------------------------------------#
    def test_Vanilla(self):
        arch = Vanilla(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = Vanilla(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4, channels=3, architecture="2D.Vanilla", input_resolution=(32, 32)
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["Vanilla"] == "z-score")
        self.assertTrue(sdm_global["2D.Vanilla"] == "z-score")

    # -------------------------------------------------#
    #              Architecture: ResNet50             #
    # -------------------------------------------------#
    def test_ResNet50(self):
        arch = ResNet50(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ResNet50(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ResNet50",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50"] == "torch")
        self.assertTrue(sdm_global["2D.ResNet50"] == "torch")

    # -------------------------------------------------#
    #             Architecture: ResNet101             #
    # -------------------------------------------------#
    def test_ResNet101(self):
        arch = ResNet101(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ResNet101(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ResNet101",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101"] == "torch")
        self.assertTrue(sdm_global["2D.ResNet101"] == "torch")

    # -------------------------------------------------#
    #             Architecture: ResNet152             #
    # -------------------------------------------------#
    def test_ResNet152(self):
        arch = ResNet152(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ResNet152(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ResNet152",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152"] == "torch")
        self.assertTrue(sdm_global["2D.ResNet152"] == "torch")

    # -------------------------------------------------#
    #              Architecture: ResNeXt50             #
    # -------------------------------------------------#
    def test_ResNeXt50(self):
        arch = ResNeXt50(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ResNeXt50(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ResNeXt50",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt50"] == "torch")
        self.assertTrue(sdm_global["2D.ResNeXt50"] == "torch")

    # -------------------------------------------------#
    #              Architecture: ResNeXt101            #
    # -------------------------------------------------#
    def test_ResNeXt101(self):
        arch = ResNeXt101(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ResNeXt101(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ResNeXt101",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt101"] == "torch")
        self.assertTrue(sdm_global["2D.ResNeXt101"] == "torch")

    # -------------------------------------------------#
    #            Architecture: DenseNet121            #
    # -------------------------------------------------#
    def test_DenseNet121(self):
        arch = DenseNet121(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = DenseNet121(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.DenseNet121",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet121"] == "torch")
        self.assertTrue(sdm_global["2D.DenseNet121"] == "torch")

    # -------------------------------------------------#
    #            Architecture: DenseNet169            #
    # -------------------------------------------------#
    def test_DenseNet169(self):
        arch = DenseNet169(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = DenseNet169(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.DenseNet169",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet169"] == "torch")
        self.assertTrue(sdm_global["2D.DenseNet169"] == "torch")

    # -------------------------------------------------#
    #            Architecture: DenseNet201            #
    # -------------------------------------------------#
    def test_DenseNet201(self):
        arch = DenseNet201(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = DenseNet201(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.DenseNet201",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet201"] == "torch")
        self.assertTrue(sdm_global["2D.DenseNet201"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB0          #
    # -------------------------------------------------#
    def test_EfficientNetB0(self):
        arch = EfficientNetB0(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB0(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB0",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB0"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB0"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB1          #
    # -------------------------------------------------#
    def test_EfficientNetB1(self):
        arch = EfficientNetB1(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB1(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB1",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB1"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB1"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB2          #
    # -------------------------------------------------#
    def test_EfficientNetB2(self):
        arch = EfficientNetB2(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB2(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB2",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB2"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB2"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB3          #
    # -------------------------------------------------#
    def test_EfficientNetB3(self):
        arch = EfficientNetB3(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB3(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB3",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB3"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB3"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB4          #
    # -------------------------------------------------#
    def test_EfficientNetB4(self):
        arch = EfficientNetB4(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB4(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB4",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB4"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB4"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB5          #
    # -------------------------------------------------#
    def test_EfficientNetB5(self):
        arch = EfficientNetB5(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB5(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB5",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB5"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB5"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB6          #
    # -------------------------------------------------#
    def test_EfficientNetB6(self):
        arch = EfficientNetB6(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB6(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB6",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB6"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB6"] == "torch")

    # -------------------------------------------------#
    #           Architecture: EfficientNetB7          #
    # -------------------------------------------------#
    def test_EfficientNetB7(self):
        arch = EfficientNetB7(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = EfficientNetB7(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.EfficientNetB7",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["EfficientNetB7"] == "torch")
        self.assertTrue(sdm_global["2D.EfficientNetB7"] == "torch")

    # -------------------------------------------------#
    #            Architecture: MobileNetV2            #
    # -------------------------------------------------#
    def test_MobileNetV2(self):
        arch = MobileNetV2(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = MobileNetV2(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.MobileNetV2",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNetV2"] == "torch")
        self.assertTrue(sdm_global["2D.MobileNetV2"] == "torch")

    # -------------------------------------------------#
    #            Architecture: InceptionV3            #
    # -------------------------------------------------#
    def test_InceptionV3(self):
        self.dataloader_GRAY.sf_resize = Resize(shape=(75, 75))
        self.dataloader_RGB.sf_resize = Resize(shape=(75, 75))
        arch = InceptionV3(channels=1, input_resolution=(75, 75))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = InceptionV3(channels=3, input_resolution=(75, 75))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.InceptionV3",
            input_resolution=(75, 75),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["InceptionV3"] == "torch")
        self.assertTrue(sdm_global["2D.InceptionV3"] == "torch")
        self.dataloader_GRAY.sf_resize = Resize(shape=(32, 32))
        self.dataloader_RGB.sf_resize = Resize(shape=(32, 32))

    # -------------------------------------------------#
    #               Architecture: VGG16               #
    # -------------------------------------------------#
    def test_VGG16(self):
        arch = VGG16(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = VGG16(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4, channels=3, architecture="2D.VGG16", input_resolution=(32, 32)
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["VGG16"] == "torch")
        self.assertTrue(sdm_global["2D.VGG16"] == "torch")

    # -------------------------------------------------#
    #               Architecture: VGG19               #
    # -------------------------------------------------#
    def test_VGG19(self):
        arch = VGG19(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = VGG19(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4, channels=3, architecture="2D.VGG19", input_resolution=(32, 32)
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["VGG19"] == "torch")
        self.assertTrue(sdm_global["2D.VGG19"] == "torch")

    # -------------------------------------------------#
    #              Architecture: ViT B16              #
    # -------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    def test_ViT_B16(self):
        self.dataloader_RGB.sf_resize = Resize(shape=(224, 224))
        arch = ViT_B16(channels=3, input_resolution=(224, 224))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ViT_B16",
            input_resolution=(224, 224),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ViT_B16"] == "torch")
        self.assertTrue(sdm_global["2D.ViT_B16"] == "torch")
        self.assertTrue("2D.ViT_B16" in architecture_dict)
        self.dataloader_RGB.sf_resize = Resize(shape=(32, 32))

    # -------------------------------------------------#
    #              Architecture: ViT B32              #
    # -------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    # def test_ViT_B32(self):
    # self.dataloader_RGB.sf_resize = Resize(shape=(224, 224))
    # arch = ViT_B32(channels=3, input_resolution=(224, 224))
    # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
    # model.predict(self.dataloader_RGB)
    # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_B32",
    #                        input_resolution=(224, 224))
    # try : print(model.model)
    # except : raise Exception()
    # self.assertTrue(supported_standardize_mode["ViT_B32"] == "torch")
    # self.assertTrue(sdm_global["2D.ViT_B32"] == "torch")
    # self.assertTrue("2D.ViT_B32" in architecture_dict)
    # self.dataloader_RGB.sf_resize = Resize(shape=(32, 32))

    # -------------------------------------------------#
    #              Architecture: ViT L16              #
    # -------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    # def test_ViT_L16(self):
    # self.dataloader_RGB.sf_resize = Resize(shape=(384, 384))
    # arch = ViT_L16(channels=3, input_resolution=(384, 384))
    # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
    # model.predict(self.dataloader_RGB)
    # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_L16",
    #                        input_resolution=(384, 384))
    # try : print(model.model)
    # except : raise Exception()
    # self.assertTrue(supported_standardize_mode["ViT_L16"] == "torch")
    # self.assertTrue(sdm_global["2D.ViT_L16"] == "torch")
    # self.assertTrue("2D.ViT_L16" in architecture_dict)
    # self.dataloader_RGB.sf_resize = Resize(shape=(32, 32))

    # -------------------------------------------------#
    #              Architecture: ViT L32              #
    # -------------------------------------------------#
    # Functionality and Interoperability testing deactived due to too intensive RAM requirements
    # def test_ViT_L32(self):
    # self.dataloader_RGB.sf_resize = Resize(shape=(384, 384))
    # arch = ViT_L32(channels=3, input_resolution=(384, 384))
    # model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
    # model.predict(self.dataloader_RGB)
    # model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ViT_L32",
    #                        input_resolution=(384, 384))
    # try : print(model.model)
    # except : raise Exception()
    # self.assertTrue(supported_standardize_mode["ViT_L32"] == "torch")
    # self.assertTrue(sdm_global["2D.ViT_L32"] == "torch")
    # self.assertTrue("2D.ViT_L32" in architecture_dict)
    # self.dataloader_RGB.sf_resize = Resize(shape=(32, 32))

    # -------------------------------------------------#
    #            Architecture: ConvNeXtBase           #
    # -------------------------------------------------#
    def test_ConvNeXtBase(self):
        arch = ConvNeXtBase(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ConvNeXtBase(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ConvNeXtBase",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtBase"] == "torch")
        self.assertTrue(sdm_global["2D.ConvNeXtBase"] == "torch")

    # -------------------------------------------------#
    #            Architecture: ConvNeXtTiny           #
    # -------------------------------------------------#
    def test_ConvNeXtTiny(self):
        arch = ConvNeXtTiny(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ConvNeXtTiny(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ConvNeXtTiny",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtTiny"] == "torch")
        self.assertTrue(sdm_global["2D.ConvNeXtTiny"] == "torch")

    # -------------------------------------------------#
    #           Architecture: ConvNeXtSmall           #
    # -------------------------------------------------#
    def test_ConvNeXtSmall(self):
        arch = ConvNeXtSmall(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ConvNeXtSmall(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ConvNeXtSmall",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtSmall"] == "torch")
        self.assertTrue(sdm_global["2D.ConvNeXtSmall"] == "torch")

    # -------------------------------------------------#
    #           Architecture: ConvNeXtLarge           #
    # -------------------------------------------------#
    def test_ConvNeXtLarge(self):
        arch = ConvNeXtLarge(channels=1, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_GRAY)
        arch = ConvNeXtLarge(channels=3, input_resolution=(32, 32))
        model = NeuralNetwork(n_labels=4, channels=3, architecture=arch)
        model.predict(self.dataloader_RGB)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="2D.ConvNeXtLarge",
            input_resolution=(32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtLarge"] == "torch")
        self.assertTrue(sdm_global["2D.ConvNeXtLarge"] == "torch")
