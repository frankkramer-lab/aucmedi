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
class ArchitecturesImageTEST(unittest.TestCase):
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
            img_rgb = np.random.rand(16, 16, 8, 3) * 255
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
