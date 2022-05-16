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
from aucmedi import Neural_Network
from aucmedi.data_processing.io_loader import numpy_loader
from aucmedi.ensemble import *

#-----------------------------------------------------#
#                 Unittest: Ensembler                 #
#-----------------------------------------------------#
class EnsemblerTEST(unittest.TestCase):
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
        # Create 3D data
        self.sampleList3D = []
        for i in range(0, 3):
            img_gray = np.random.rand(16, 16, 16) * 255
            index = "image.sample_" + str(i) + ".GRAY.npy"
            path_sampleGRAY = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleGRAY, img_gray)
            self.sampleList3D.append(index)
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
    def test_Augmenting_2D(self):
        preds = predict_augmenting(model=self.model2D, samples=self.sampleList2D,
                                   path_imagedir=self.tmp_data.name,
                                   n_cycles=1, batch_size=10,
                                   data_aug=None, image_format=None,
                                   resize=None, grayscale=False,
                                   subfunctions=[], standardize_mode="tf",
                                   seed=None, workers=1)
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))
        preds = predict_augmenting(model=self.model2D, samples=self.sampleList2D,
                                   path_imagedir=self.tmp_data.name,
                                   n_cycles=5, batch_size=10,
                                   data_aug=None, image_format=None,
                                   resize=None, grayscale=False,
                                   subfunctions=[], standardize_mode="tf",
                                   seed=None, workers=1)
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))

    def test_Augmenting_3D(self):
        preds = predict_augmenting(model=self.model3D, samples=self.sampleList3D,
                                   path_imagedir=self.tmp_data.name,
                                   n_cycles=1, batch_size=3,
                                   data_aug=None, image_format=None,
                                   resize=None, grayscale=True, two_dim=False,
                                   subfunctions=[], standardize_mode="tf",
                                   seed=None, workers=1, loader=numpy_loader)
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))
        preds = predict_augmenting(model=self.model3D, samples=self.sampleList3D,
                                   path_imagedir=self.tmp_data.name,
                                   n_cycles=5, batch_size=8,
                                   data_aug=None, image_format=None,
                                   resize=None, grayscale=True, two_dim=False,
                                   subfunctions=[], standardize_mode="tf",
                                   seed=None, workers=1, loader=numpy_loader)
        self.assertTrue(np.array_equal(preds.shape, (3, 4)))
