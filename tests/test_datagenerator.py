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
import shutil
#Internal libraries
from aucmedi import DataGenerator

#-----------------------------------------------------#
#               Unittest: Data Generator              #
#-----------------------------------------------------#
class DataGeneratorTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create Grayscale data
        self.sampleList_gray = []
        for i in range(0, 25):
            img_gray = np.random.rand(16, 16) * 255
            imgGRAY_pillow = Image.fromarray(img_gray.astype(np.uint8))
            index = "image.sample_" + str(i) + ".GRAY.png"
            path_sampleGRAY = os.path.join(self.tmp_data.name, index)
            imgGRAY_pillow.save(path_sampleGRAY)
            self.sampleList_gray.append(index)

        # Create RGB data
        self.sampleList_rgb = []
        for i in range(0, 25):
            img_rgb = np.random.rand(16, 16, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((25, 4), dtype=np.uint8)
        for i in range(0, 25):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

    #-------------------------------------------------#
    #           Initialization Functionality          #
    #-------------------------------------------------#
    # Class Creation
    def test_BASE_create(self):
        data_gen = DataGenerator(self.sampleList_rgb, self.tmp_data.name)
        self.assertIsInstance(data_gen, DataGenerator)

    #-------------------------------------------------#
    #            Application Functionality            #
    #-------------------------------------------------#
    # Usage: Grayscale without Labels
    def test_RUN_GRAYSCALE_noLabel(self):
        data_gen = DataGenerator(self.sampleList_gray, self.tmp_data.name,
                                 grayscale=True, batch_size=5)
        for i in range(0, 10):
            batch = next(data_gen)
            self.assertTrue(len(batch), 1)
            self.assertTrue(np.array_equal(batch[0].shape, (5, 224, 224, 1)))

    # Usage: RGB without Labels
    def test_RUN_RGB_noLabel(self):
        data_gen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                 grayscale=False, batch_size=5)
        for i in range(0, 10):
            batch = next(data_gen)
            self.assertTrue(len(batch), 1)
            self.assertTrue(np.array_equal(batch[0].shape, (5, 224, 224, 3)))

    # Usage: With Labels
    def test_RUN_withLabel(self):
        data_gen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                 labels=self.labels_ohe,
                                 grayscale=False, batch_size=5)
        for i in range(0, 10):
            batch = next(data_gen)
            self.assertTrue(len(batch), 2)
            self.assertTrue(np.array_equal(batch[1].shape, (5, 4)))

    #-------------------------------------------------#
    #                 Multi-Processing                #
    #-------------------------------------------------#
    def test_MP(self):
        data_gen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                 labels=self.labels_ohe,
                                 grayscale=False, batch_size=5, workers=5)
        for i in range(0, 10):
            batch = next(data_gen)
            self.assertTrue(len(batch), 2)
            self.assertTrue(np.array_equal(batch[1].shape, (5, 4)))

    #-------------------------------------------------#
    #             Beforehand Preprocessing            #
    #-------------------------------------------------#
    def test_PrepareImages(self):
        data_gen = DataGenerator(self.sampleList_rgb, self.tmp_data.name,
                                 labels=self.labels_ohe, prepare_images=True,
                                 grayscale=False, batch_size=5)
        precprocessed_images = os.listdir(data_gen.prepare_dir)
        self.assertTrue(len(precprocessed_images), len(self.sampleList_rgb))
        for i in range(0, 10):
            batch = next(data_gen)
            self.assertTrue(len(batch), 2)
            self.assertTrue(np.array_equal(batch[1].shape, (5, 4)))
        shutil.rmtree(data_gen.prepare_dir)
