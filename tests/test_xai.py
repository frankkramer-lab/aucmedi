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
import tempfile
import os
from PIL import Image
import numpy as np
#Internal libraries
from aucmedi import *
from aucmedi.xai import *
from aucmedi.xai.methods import *

#-----------------------------------------------------#
#              Unittest: Explainable AI               #
#-----------------------------------------------------#
class xaiTEST(unittest.TestCase):
    # Setup AUCMEDI pipeline
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create RGB data
        self.sampleList = []
        for i in range(0, 10):
            img_rgb = np.random.rand(32, 32, 3) * 255
            img_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sample = os.path.join(self.tmp_data.name, index)
            img_pillow.save(path_sample)
            self.sampleList.append(index)
        # Create classification labels
        self.labels_ohe = np.zeros((10, 4), dtype=np.uint8)
        for i in range(0, 10):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create Data Generator
        self.datagen = DataGenerator(self.sampleList,  self.tmp_data.name,
                                     labels=self.labels_ohe, resize=None,
                                     grayscale=False, batch_size=3)
        # Create Neural Network model
        self.model = Neural_Network(n_labels=4, channels=3, input_shape=(32,32),
                                    architecture="Vanilla", batch_queue_size=1)
        # Compute predictions
        self.preds = self.model.predict(self.datagen)
        # Initialize testing image
        self.image = next(self.datagen)[0][[0]]

    #-------------------------------------------------#
    #             XAI Functions: Decoder              #
    #-------------------------------------------------#
    def test_Decoder_argmax(self):
        imgs, hms = xai_decoder(self.datagen, self.model, preds=self.preds)
        self.assertTrue(np.array_equal(imgs.shape, (10, 32, 32, 3)))
        self.assertTrue(np.array_equal(hms.shape, (10, 32, 32)))

    def test_Decoder_allclasses(self):
        imgs, hms = xai_decoder(self.datagen, self.model, preds=None)
        self.assertTrue(np.array_equal(imgs.shape, (10, 32, 32, 3)))
        self.assertTrue(np.array_equal(hms.shape, (10, 4, 32, 32)))

    #-------------------------------------------------#
    #              XAI Methods: Grad-Cam              #
    #-------------------------------------------------#
    def test_XAImethod_GradCam_init(self):
        GradCAM(self.model.model)
        xai_dict["gradcam"](self.model.model)

    def test_XAImethod_GradCam_heatmap(self):
        xai_method = GradCAM(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (2,2)))

    def test_XAImethod_GradCam_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="gradcam")
        self.assertTrue(np.array_equal(hms.shape, (10, 4, 32, 32)))
