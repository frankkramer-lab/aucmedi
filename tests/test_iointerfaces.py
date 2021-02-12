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
import tempfile
from PIL import Image
import os
#Internal libraries
from aucmedi.data_processing.io_interfaces import *

#-----------------------------------------------------#
#               Unittest: IO Interfaces               #
#-----------------------------------------------------#
class IOinterfacesTEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        self.aif = ["png"]

    #-------------------------------------------------#
    #              Directory IO Interface             #
    #-------------------------------------------------#
    def test_Directory_testing(self):
        # Create imaging data
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 25):
            img = np.random.rand(16, 16, 3) * 255
            img_pillow = Image.fromarray(img.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            path_sample = os.path.join(tmp_data.name, index)
            img_pillow.save(path_sample)
        # Run Directory IO
        ds = directory_loader(tmp_data.name, self.aif, training=False)
        self.assertTrue(len(ds[0]), 25)

    def test_Directory_training(self):
        # Create imaging data with subdirectories
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            os.mkdir(os.path.join(tmp_data.name, "class_" + str(i)))
        # Fill subdirectories with images
        for i in range(0, 25):
            img = np.random.rand(16, 16, 3) * 255
            img_pillow = Image.fromarray(img.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            label_dir = "class_" + str((i % 5))
            path_sample = os.path.join(tmp_data.name, label_dir, index)
            img_pillow.save(path_sample)
        # Run Directory IO
        ds = directory_loader(tmp_data.name, self.aif, training=True)
        self.assertTrue(len(ds[0]), 25)
