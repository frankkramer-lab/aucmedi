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
import numpy as np
#Internal libraries
from aucmedi.data_processing.subfunctions import *

#-----------------------------------------------------#
#               Unittest: Subfunctions                #
#-----------------------------------------------------#
class SubfunctionsTEST(unittest.TestCase):
    # Create random imaging data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create Grayscale data
        img_gray = np.random.rand(16, 24, 1) * 255
        self.imgGRAY = np.float32(img_gray)
        # Create RGB data
        img_rgb = np.random.rand(16, 24, 3) * 255
        self.imgRGB = np.float32(img_rgb)

    #-------------------------------------------------#
    #              Subfunction: Padding               #
    #-------------------------------------------------#
    def test_PADDING_create(self):
        sf = Padding()

    def test_PADDING_transform(self):
        sf = Padding(shape=(32, 32), mode="edge")
        img_ppGRAY = sf.transform(self.imgGRAY.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 32, 1)))
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 32, 3)))
        sf = Padding(shape=(8, 32), mode="constant")
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (16, 32, 3)))
        sf = Padding(mode="square")
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (24, 24, 3)))

    #-------------------------------------------------#
    #               Subfunction: Resize               #
    #-------------------------------------------------#
    def test_RESIZE_create(self):
        sf = Resize()

    def test_RESIZE_transform(self):
        sf = Resize(shape=(32, 32))
        img_ppGRAY = sf.transform(self.imgGRAY.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 32, 1)))
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 32, 3)))
        sf = Resize(shape=(8, 8))
        img_ppGRAY = sf.transform(self.imgGRAY.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (8, 8, 1)))
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (8, 8, 3)))

    #-------------------------------------------------#
    #             Subfunction: Standardize            #
    #-------------------------------------------------#
    def test_STANDARDIZE_create(self):
        sf = Standardize()

    def test_STANDARDIZE_transform(self):
        sf = Standardize(mode="tf")
        img_ppGRAY = sf.transform(self.imgGRAY.copy())
        self.assertTrue(np.amin(img_ppGRAY) < 0)
        self.assertTrue(np.amax(img_ppGRAY) > 0)
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.amin(img_ppRGB) < 0)
        self.assertTrue(np.amax(img_ppRGB) > 0)
        sf = Standardize(mode="caffe")
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.amin(img_ppRGB) < 0)
        self.assertTrue(np.amax(img_ppRGB) > 0)
        sf = Standardize(mode="torch")
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.amin(img_ppRGB) < 0)
        self.assertTrue(np.amax(img_ppRGB) > 0)

    #-------------------------------------------------#
    #              Subfunction: Cropping              #
    #-------------------------------------------------#
    def test_CROP_create(self):
        sf = Crop()

    def test_CROP_transform(self):
        sf = Crop(shape=(16, 12))
        img_ppGRAY = sf.transform(self.imgGRAY.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (16, 12, 1)))
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (16, 12, 3)))
        sf = Crop(shape=(8, 8))
        img_ppGRAY = sf.transform(self.imgGRAY.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (8, 8, 1)))
        img_ppRGB = sf.transform(self.imgRGB.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (8, 8, 3)))
