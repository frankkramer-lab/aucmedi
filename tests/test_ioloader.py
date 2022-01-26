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
import numpy as np
import tempfile
from PIL import Image
import os
#Internal libraries
from aucmedi.data_processing.io_loader import *

#-----------------------------------------------------#
#                 Unittest: IO Loader                 #
#-----------------------------------------------------#
class IOloaderTEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        self.img_2d_gray = np.random.rand(16, 16, 1) * 255
        self.img_2d_rgb = np.random.rand(16, 16, 3) * 255
        self.img_3d_gray = np.random.rand(16, 16, 16, 1) * 255
        self.img_3d_rgb = np.random.rand(16, 16, 16, 3) * 255

    #-------------------------------------------------#
    #                  Image Loader                   #
    #-------------------------------------------------#
    # Test for grayscale images
    def test_image_loader_2Dgray(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            # Create image
            img_store = np.squeeze(self.img_2d_gray, axis=-1)
            img_pillow = Image.fromarray(img_store.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            path_sample = os.path.join(tmp_data.name, index)
            img_pillow.save(path_sample)
            # Load image via loader
            img = image_loader(index, tmp_data.name, image_format=None,
                               grayscale=True)
            self.assertTrue(np.array_equal(img.shape, self.img_2d_gray.shape))

    # Test for rgb images
    def test_image_loader_2Drgb(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            # Create image
            img_pillow = Image.fromarray(self.img_2d_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            path_sample = os.path.join(tmp_data.name, index)
            img_pillow.save(path_sample)
            # Load image via loader
            img = image_loader(index, tmp_data.name, image_format=None,
                               grayscale=False)
            self.assertTrue(np.array_equal(img.shape, self.img_2d_rgb.shape))
