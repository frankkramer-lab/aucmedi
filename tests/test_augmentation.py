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
from aucmedi import Image_Augmentation

#-----------------------------------------------------#
#             Unittest: Image Augmentation            #
#-----------------------------------------------------#
class AugmentationTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create Grayscale data
        img_gray = np.random.rand(16, 16, 1) * 254
        self.imgGRAY = np.float32(img_gray)
        # Create RGB data
        img_rgb = np.random.rand(16, 16, 3) * 254
        self.imgRGB = np.float32(img_rgb)

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_BASE_create(self):
        data_aug = Image_Augmentation()
        self.assertIsInstance(data_aug, Image_Augmentation)

    # Application
    def test_BASE_application(self):
        data_aug = Image_Augmentation()
        img_augGRAY = data_aug.apply(self.imgGRAY)
        img_augRGB= data_aug.apply(self.imgRGB)
        self.assertFalse(np.array_equal(img_augGRAY, self.imgGRAY))
        self.assertFalse(np.array_equal(img_augRGB, self.imgRGB))

    # Rebuild Augmentation Operator
    def test_BASE_rebuild(self):
        data_aug = Image_Augmentation(flip=False, rotate=False,
                     brightness=False, contrast=False, saturation=False,
                     hue=False, scale=False, crop=False, grid_distortion=False,
                     compression=False, gaussian_noise=False,
                     gaussian_blur=False, downscaling=False, gamma=False,
                     elastic_transform=False)
        img_augRGB = data_aug.apply(self.imgRGB)
        self.assertTrue(np.array_equal(img_augRGB, self.imgRGB))
        data_aug.aug_flip = True
        data_aug.aug_flip_p = 1.0
        data_aug.build()
        img_augRGB = data_aug.apply(self.imgRGB)
        self.assertFalse(np.array_equal(img_augRGB, self.imgRGB))
