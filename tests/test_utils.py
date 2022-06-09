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
from PIL import Image
import tempfile
import os
from tensorflow.keras.callbacks import CSVLogger
#Internal libraries
from aucmedi.utils.callbacks import *
from aucmedi import *

#-----------------------------------------------------#
#                  Unittest: Utility                  #
#-----------------------------------------------------#
class UtilityTEST(unittest.TestCase):
    # Create random imaging data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")

        # Create RGB data
        self.sampleList_rgb = []
        for i in range(0, 1):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((1, 4), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create RGB Data Generator
        self.datagen = DataGenerator(self.sampleList_rgb,
                                     self.tmp_data.name,
                                     labels=self.labels_ohe,
                                     resize=(32, 32),
                                     grayscale=False, batch_size=1)

    #-------------------------------------------------#
    #             Callbacks: CSV2history              #
    #-------------------------------------------------#
    def test_Callbacks_csv2history(self):
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".model")
        path_csv = os.path.join(self.tmp_data.name, "testing.csv")
        csvlog = CSVLogger(path_csv)

        model = NeuralNetwork(n_labels=4, channels=3, batch_queue_size=1)
        hist_returned = model.train(training_generator=self.datagen,
                                    validation_generator=self.datagen,
                                    epochs=3, callbacks=[csvlog])

        hist_loaded = csv_to_history(path_csv)
        del hist_loaded["epoch"]

        for key in hist_returned:
            self.assertTrue(key in hist_returned and key in hist_loaded)
            self.assertTrue(len(hist_returned[key]) == len(hist_loaded[key]))
