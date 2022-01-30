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
from aucmedi import DataGenerator

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
    # Test for DataGenerator functionality
    def test_image_loader_DataGenerator(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        # Create dataset
        sample_list = []
        for i in range(0, 6):
           img_pillow = Image.fromarray(self.img_2d_rgb.astype(np.uint8))
           index = "image.sample_" + str(i) + ".png"
           path_sample = os.path.join(tmp_data.name, index)
           img_pillow.save(path_sample)
           sample_list.append(index)
        # Test DataGenerator
        data_gen = DataGenerator(sample_list, tmp_data.name, resize=None,
                                 grayscale=False, batch_size=2)
        for i in range(0, 3):
            batch = next(data_gen)
            self.assertTrue(np.array_equal(batch[0].shape, (2, 16, 16, 3)))

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

    #-------------------------------------------------#
    #                  NumPy Loader                   #
    #-------------------------------------------------#
    # Test for DataGenerator functionality
    def test_numpy_loader_DataGenerator(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        # Create dataset
        sample_list = []
        for i in range(0, 6):
           index = "3Dimage.sample_" + str(i) + ".npy"
           path_sample = os.path.join(tmp_data.name, index)
           np.save(path_sample, self.img_3d_gray)
           sample_list.append(index)
        # Test DataGenerator
        data_gen = DataGenerator(sample_list, tmp_data.name, loader=numpy_loader,
                                 resize=None, two_dim=False, standardize_mode=None,
                                 grayscale=True, batch_size=2)
        for i in range(0, 3):
            batch = next(data_gen)
            self.assertTrue(np.array_equal(batch[0].shape, (2, 16, 16, 16, 1)))

    # Test for grayscale 2D images
    def test_numpy_loader_2Dgray(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            # Create image
            index = "image.sample_" + str(i) + ".npy"
            path_sample = os.path.join(tmp_data.name, index)
            np.save(path_sample, self.img_2d_gray)
            # Load image via loader
            img = numpy_loader(index, tmp_data.name, image_format=None,
                               grayscale=True, two_dim=True)
            self.assertTrue(np.array_equal(img.shape, self.img_2d_gray.shape))

    # Test for grayscale 3D images
    def test_numpy_loader_3Dgray(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            # Create image
            index = "image.sample_" + str(i) + ".npy"
            path_sample = os.path.join(tmp_data.name, index)
            np.save(path_sample, self.img_3d_gray)
            # Load image via loader
            img = numpy_loader(index, tmp_data.name, image_format=None,
                               grayscale=True, two_dim=False)
            self.assertTrue(np.array_equal(img.shape, self.img_3d_gray.shape))

    # Test for rgb 2D images
    def test_numpy_loader_2Drgb(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            # Create image
            index = "image.sample_" + str(i) + ".npy"
            path_sample = os.path.join(tmp_data.name, index)
            np.save(path_sample, self.img_2d_rgb)
            # Load image via loader
            img = numpy_loader(index, tmp_data.name, image_format=None,
                               grayscale=False, two_dim=True)
            self.assertTrue(np.array_equal(img.shape, self.img_2d_rgb.shape))

    # Test for rgb 3D images
    def test_numpy_loader_3Drgb(self):
        # Create temporary directory
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            # Create image
            index = "image.sample_" + str(i) + ".npy"
            path_sample = os.path.join(tmp_data.name, index)
            np.save(path_sample, self.img_3d_rgb)
            # Load image via loader
            img = numpy_loader(index, tmp_data.name, image_format=None,
                               grayscale=False, two_dim=False)
            self.assertTrue(np.array_equal(img.shape, self.img_3d_rgb.shape))
