# ==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2026 IT-Infrastructure for Translational Medical Research,    #
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
import numpy as np

# Internal libraries
from aucmedi.neural_network.architectures.volume import *
from aucmedi.neural_network.architectures import (
    supported_standardize_mode as sdm_global,
)
from aucmedi import *
from aucmedi.data_processing.io_loader import numpy_loader


# -----------------------------------------------------#
#               Unittest: Architectures               #
# -----------------------------------------------------#
class ArchitecturesVolumeTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )
        # Create Houndsfield Units data
        self.sampleList_hu = []
        for i in range(0, 1):
            img_hu = (np.random.rand(32, 32, 32) * 2000) - 500
            index = "image.sample_" + str(i) + ".HU.npy"
            path_sampleHU = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleHU, img_hu)
            self.sampleList_hu.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((1, 4), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create HU Data Loader
        self.dataloader_HU = create_batch_loader(
            self.sampleList_hu,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(32, 32, 32),
            loader=numpy_loader,
            two_dim=False,
            grayscale=True,
            batch_size=1,
        )

        self.dataloader_HU_64 = create_batch_loader(
            self.sampleList_hu,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(64, 64, 64),
            loader=numpy_loader,
            two_dim=False,
            grayscale=True,
            batch_size=1,
        )

    # -------------------------------------------------#
    #              Architecture: Vanilla              #
    # -------------------------------------------------#
    def test_Vanilla(self):
        print("------------------------ SHAPE of HU:", self.sampleList_hu)
        arch = Vanilla(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.Vanilla",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["Vanilla"] == "z-score")
        self.assertTrue(sdm_global["3D.Vanilla"] == "z-score")

    # -------------------------------------------------#
    #            Architecture: DenseNet121            #
    # -------------------------------------------------#
    def test_DenseNet121(self):
        arch = DenseNet121(channels=1, input_resolution=(64, 64, 64))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU_64)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.DenseNet121",
            input_resolution=(64, 64, 64),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet121"] == "torch")
        self.assertTrue(sdm_global["3D.DenseNet121"] == "torch")

    # -------------------------------------------------#
    #            Architecture: DenseNet169            #
    # -------------------------------------------------#
    def test_DenseNet169(self):
        arch = DenseNet169(channels=1, input_resolution=(64, 64, 64))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU_64)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.DenseNet169",
            input_resolution=(64, 64, 64),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet169"] == "torch")
        self.assertTrue(sdm_global["3D.DenseNet169"] == "torch")

    # -------------------------------------------------#
    #            Architecture: DenseNet201            #
    # -------------------------------------------------#
    def test_DenseNet201(self):
        arch = DenseNet201(channels=1, input_resolution=(64, 64, 64))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU_64)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.DenseNet201",
            input_resolution=(64, 64, 64),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["DenseNet201"] == "torch")
        self.assertTrue(sdm_global["3D.DenseNet201"] == "torch")

    # -------------------------------------------------#
    #              Architecture: ResNet18             #
    # -------------------------------------------------#
    def test_ResNet18(self):
        arch = ResNet18(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNet18",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet18"] == "torch")
        self.assertTrue(sdm_global["3D.ResNet18"] == "torch")

    # -------------------------------------------------#
    #              Architecture: ResNet34             #
    # -------------------------------------------------#
    def test_ResNet34(self):
        arch = ResNet34(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNet34",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet34"] == "torch")
        self.assertTrue(sdm_global["3D.ResNet34"] == "torch")

    # -------------------------------------------------#
    #              Architecture: ResNet50             #
    # -------------------------------------------------#
    def test_ResNet50(self):
        arch = ResNet50(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNet50",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet50"] == "torch")
        self.assertTrue(sdm_global["3D.ResNet50"] == "torch")

    # -------------------------------------------------#
    #             Architecture: ResNet101             #
    # -------------------------------------------------#
    def test_ResNet101(self):
        arch = ResNet101(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNet101",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet101"] == "torch")
        self.assertTrue(sdm_global["3D.ResNet101"] == "torch")

    # -------------------------------------------------#
    #             Architecture: ResNet152             #
    # -------------------------------------------------#
    def test_ResNet152(self):
        arch = ResNet152(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNet152",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNet152"] == "torch")
        self.assertTrue(sdm_global["3D.ResNet152"] == "torch")

    # -------------------------------------------------#
    #             Architecture: ResNeXt50             #
    # -------------------------------------------------#
    def test_ResNeXt50(self):
        arch = ResNeXt50(channels=1, input_resolution=(64, 64, 64))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU_64)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNeXt50",
            input_resolution=(64, 64, 64),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt50"] == "torch")
        self.assertTrue(sdm_global["3D.ResNeXt50"] == "torch")

    # -------------------------------------------------#
    #            Architecture: ResNeXt101             #
    # -------------------------------------------------#
    def test_ResNeXt101(self):
        arch = ResNeXt101(channels=1, input_resolution=(64, 64, 64))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU_64)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ResNeXt101",
            input_resolution=(64, 64, 64),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ResNeXt101"] == "torch")
        self.assertTrue(sdm_global["3D.ResNeXt101"] == "torch")

    # -------------------------------------------------#
    #               Architecture: VGG16               #
    # -------------------------------------------------#
    def test_VGG16(self):
        arch = VGG16(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.VGG16",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["VGG16"] == "torch")
        self.assertTrue(sdm_global["3D.VGG16"] == "torch")

    # -------------------------------------------------#
    #               Architecture: VGG19               #
    # -------------------------------------------------#
    def test_VGG19(self):
        arch = VGG19(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.VGG19",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["VGG19"] == "torch")
        self.assertTrue(sdm_global["3D.VGG19"] == "torch")

    # -------------------------------------------------#
    #            Architecture: MobileNetV2            #
    # -------------------------------------------------#
    def test_MobileNetV2(self):
        arch = MobileNetV2(channels=1, input_resolution=(64, 64, 64))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU_64)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.MobileNetV2",
            input_resolution=(64, 64, 64),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["MobileNetV2"] == "torch")
        self.assertTrue(sdm_global["3D.MobileNetV2"] == "torch")

    # -------------------------------------------------#
    #           Architecture: ConvNeXt Base           #
    # -------------------------------------------------#
    def test_ConvNeXtBase(self):
        arch = ConvNeXtBase(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ConvNeXtBase",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtBase"] == "torch")
        self.assertTrue(sdm_global["3D.ConvNeXtBase"] == "torch")

    # -------------------------------------------------#
    #           Architecture: ConvNeXt Tiny           #
    # -------------------------------------------------#
    def test_ConvNeXtTiny(self):
        arch = ConvNeXtTiny(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ConvNeXtTiny",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtTiny"] == "torch")
        self.assertTrue(sdm_global["3D.ConvNeXtTiny"] == "torch")

    # -------------------------------------------------#
    #          Architecture: ConvNeXt Small           #
    # -------------------------------------------------#
    def test_ConvNeXtSmall(self):
        arch = ConvNeXtSmall(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ConvNeXtSmall",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtSmall"] == "torch")
        self.assertTrue(sdm_global["3D.ConvNeXtSmall"] == "torch")

    # -------------------------------------------------#
    #          Architecture: ConvNeXt Large           #
    # -------------------------------------------------#
    def test_ConvNeXtLarge(self):
        arch = ConvNeXtLarge(channels=1, input_resolution=(32, 32, 32))
        model = NeuralNetwork(n_labels=4, channels=1, architecture=arch)
        model.predict(self.dataloader_HU)
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            architecture="3D.ConvNeXtLarge",
            input_resolution=(32, 32, 32),
        )
        try:
            print(model.model)
        except:
            raise Exception()
        self.assertTrue(supported_standardize_mode["ConvNeXtLarge"] == "torch")
        self.assertTrue(sdm_global["3D.ConvNeXtLarge"] == "torch")
