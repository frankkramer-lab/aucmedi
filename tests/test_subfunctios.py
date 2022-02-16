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
        # Create 2D Grayscale data
        img_gray = np.random.rand(16, 24, 1) * 255
        self.img2Dgray = np.float32(img_gray)
        # Create 2D RGB data
        img_rgb = np.random.rand(16, 24, 3) * 255
        self.img2Drgb = np.float32(img_rgb)
        # Create 3D Grayscale data
        img_3Dgray = np.random.rand(16, 24, 32, 1) * 255
        self.img3Dgray = np.float32(img_3Dgray)
        # Create 3D RGB data
        img_3Drgb = np.random.rand(16, 24, 32, 3) * 255
        self.img3Drgb = np.float32(img_3Drgb)
        # Create 3D HU data
        img_3Dhu = np.random.rand(16, 24, 32, 1) * 1500
        self.img3Dhu = np.float32(img_3Dhu - 500)

    #-------------------------------------------------#
    #              Subfunction: Padding               #
    #-------------------------------------------------#
    def test_PADDING_create(self):
        sf = Padding()

    def test_PADDING_transform(self):
        sf = Padding(shape=(32, 32), mode="edge")
        img_ppGRAY = sf.transform(self.img2Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 32, 1)))
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 32, 3)))
        sf = Padding(shape=(8, 32), mode="constant")
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (16, 32, 3)))
        sf = Padding(mode="square")
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (24, 24, 3)))
        sf = Padding(mode="square")
        img_ppHU= sf.transform(self.img3Dhu.copy())
        self.assertTrue(np.array_equal(img_ppHU.shape, (32, 32, 32, 1)))
        sf = Padding(shape=(8, 32, 48), mode="edge")
        img_ppRGB = sf.transform(self.img3Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (16, 32, 48, 3)))

    #-------------------------------------------------#
    #               Subfunction: Resize               #
    #-------------------------------------------------#
    def test_RESIZE_create(self):
        sf = Resize()

    def test_RESIZE_transform(self):
        # 2D
        sf = Resize(shape=(32, 32))
        img_ppGRAY = sf.transform(self.img2Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 32, 1)))
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 32, 3)))
        sf = Resize(shape=(8, 8))
        img_ppGRAY = sf.transform(self.img2Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (8, 8, 1)))
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (8, 8, 3)))
        sf = Resize(shape=(32, 8))
        img_ppGRAY = sf.transform(self.img2Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 8, 1)))
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 8, 3)))
        # 3D
        sf = Resize(shape=(32, 32, 32))
        img_ppGRAY = sf.transform(self.img3Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 32, 32, 1)))
        img_ppRGB = sf.transform(self.img3Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 32, 32, 3)))
        sf = Resize(shape=(8, 8, 8))
        img_ppGRAY = sf.transform(self.img3Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (8, 8, 8, 1)))
        img_ppRGB = sf.transform(self.img3Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (8, 8, 8, 3)))
        sf = Resize(shape=(32, 8, 8))
        img_ppGRAY = sf.transform(self.img3Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (32, 8, 8, 1)))
        img_ppRGB = sf.transform(self.img3Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (32, 8, 8, 3)))

    #-------------------------------------------------#
    #             Subfunction: Standardize            #
    #-------------------------------------------------#
    def test_STANDARDIZE_create(self):
        sf = Standardize()

    def test_STANDARDIZE_transform(self):
        # Custom implementations
        for mode in ["z-score", "minmax", "grayscale"]:
            sf = Standardize(mode=mode)
            for data in [self.img2Dgray, self.img2Drgb, self.img3Dgray,
                         self.img3Drgb, self.img3Dhu]:
                img_pp = sf.transform(data.copy())
                if mode == "z-score":
                    self.assertTrue(np.amin(img_pp) <= 0)
                    self.assertTrue(np.amax(img_pp) >= 0)
                elif mode == "minmax":
                    self.assertTrue(np.amin(img_pp) >= 0)
                    self.assertTrue(np.amax(img_pp) <= 1)
                elif mode == "grayscale":
                    self.assertTrue(np.amin(img_pp) >= 0)
                    self.assertTrue(np.amax(img_pp) <= 255)
        # Keras implementations
        for mode in ["tf", "caffe", "torch"]:
            sf = Standardize(mode=mode)
            for data in [self.img2Drgb, self.img3Drgb]:
                img_pp = sf.transform(data.copy())
                if mode == "tf":
                    self.assertTrue(np.amin(img_pp) >= -1)
                    self.assertTrue(np.amax(img_pp) <= 1)
                elif mode == "caffe":
                    self.assertTrue(np.amin(img_pp) <= 0)
                    self.assertTrue(np.amax(img_pp) >= 0)
                elif mode == "torch":
                    self.assertTrue(np.amin(img_pp) <= 0)
                    self.assertTrue(np.amax(img_pp) >= 0)
            # self.assertRaises(ValueError, sf.transform, self.img3Dhu.copy())

    #-------------------------------------------------#
    #              Subfunction: Cropping              #
    #-------------------------------------------------#
    def test_CROP_create(self):
        sf = Crop()
        sf = Crop(mode="center")
        sf = Crop(mode="random")
        self.assertRaises(ValueError, Crop, mode="test")

    def test_CROP_transform(self):
        # 2D testing
        sf = Crop(shape=(16, 12), mode="center")
        img_ppGRAY = sf.transform(self.img2Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (16, 12, 1)))
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (16, 12, 3)))
        sf = Crop(shape=(8, 8), mode="random")
        img_ppGRAY = sf.transform(self.img2Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (8, 8, 1)))
        img_ppRGB = sf.transform(self.img2Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (8, 8, 3)))
        # 3D testing
        sf = Crop(shape=(14, 12, 24))
        img_ppGRAY = sf.transform(self.img3Dgray.copy())
        self.assertTrue(np.array_equal(img_ppGRAY.shape, (14, 12, 24, 1)))
        img_ppRGB = sf.transform(self.img3Drgb.copy())
        self.assertTrue(np.array_equal(img_ppRGB.shape, (14, 12, 24, 3)))

    #-------------------------------------------------#
    #          Subfunction: Color Constancy           #
    #-------------------------------------------------#
    def test_COLORCONSTANCY_create(self):
        sf = ColorConstancy()

    def test_COLORCONSTANCY_transform(self):
        sf = ColorConstancy()
        img_filtered = sf.transform(self.img2Drgb.copy())
        self.assertFalse(np.array_equal(img_filtered, self.img2Drgb))
        self.assertTrue(np.array_equal(img_filtered.shape, (16, 24, 3)))
        img_filtered = sf.transform(self.img3Drgb.copy())
        self.assertFalse(np.array_equal(img_filtered, self.img3Drgb))
        self.assertTrue(np.array_equal(img_filtered.shape, (16, 24, 32, 3)))
        self.assertRaises(ValueError, sf.transform, self.img2Dgray.copy())

    #-------------------------------------------------#
    #                Subfunction: Clip                #
    #-------------------------------------------------#
    def test_CLIP_create(self):
        sf = Clip()

    def test_CLIP_transform(self):
        sf = Clip(min=10)
        img_clipped = sf.transform(self.img3Dhu.copy())
        self.assertTrue(np.amin(img_clipped) >= 10)
        sf = Clip(max=30)
        img_clipped = sf.transform(self.img3Dhu.copy())
        self.assertTrue(np.amax(img_clipped) <= 30)
        sf = Clip(min=10, max=50)
        img_clipped = sf.transform(self.img3Dhu.copy())
        self.assertTrue(np.amin(img_clipped) >= 10)
        self.assertTrue(np.amax(img_clipped) <= 50)

    #-------------------------------------------------#
    #               Subfunction: Chromer              #
    #-------------------------------------------------#
    def test_CHROMER_create(self):
        sf = Chromer()
        sf = Chromer(target="grayscale")
        sf = Chromer(target="rgb")
        self.assertRaises(ValueError, Chromer, target="test")

    def test_CHROMER_transform(self):
        # Target Grayscale
        sf = Chromer(target="grayscale")
        img_filtered = sf.transform(self.img2Drgb.copy())
        self.assertFalse(np.array_equal(img_filtered, self.img2Drgb))
        self.assertTrue(np.array_equal(img_filtered.shape, (16, 24, 1)))
        img_filtered = sf.transform(self.img3Drgb.copy())
        self.assertFalse(np.array_equal(img_filtered, self.img3Drgb))
        self.assertTrue(np.array_equal(img_filtered.shape, (16, 24, 32, 1)))
        self.assertRaises(ValueError, sf.transform, self.img2Dgray.copy())
        # Target RGB
        sf = Chromer(target="rgb")
        img_filtered = sf.transform(self.img2Dgray.copy())
        self.assertFalse(np.array_equal(img_filtered, self.img2Dgray))
        self.assertTrue(np.array_equal(img_filtered.shape, (16, 24, 3)))
        img_filtered = sf.transform(self.img3Dgray.copy())
        self.assertFalse(np.array_equal(img_filtered, self.img3Dgray))
        self.assertTrue(np.array_equal(img_filtered.shape, (16, 24, 32, 3)))
        self.assertRaises(ValueError, sf.transform, self.img3Dhu.copy())
        self.assertRaises(ValueError, sf.transform, self.img2Drgb.copy())
