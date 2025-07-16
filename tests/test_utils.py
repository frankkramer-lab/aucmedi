#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
from aucmedi.utils.visualizer import *
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
        # Create Grayscale data for 2D
        self.img_gray_2D = np.random.rand(16, 16, 1) * 255
        # Create RGB data for 2D
        self.sampleList_rgb_2D = []
        for i in range(0, 5):
            img_rgb = np.random.rand(16, 16, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb_2D.append(index)
        self.img_rgb_2D = np.random.rand(16, 16, 3) * 255
        # Create Grayscale data for 3D
        self.img_gray_3D = np.random.rand(16, 16, 16, 1) * 255
        # Create HU data for 3D
        self.img_hu_3D = (np.random.rand(16, 16, 16, 1) * 1000) - 500
        # Create RGB data for 3D
        self.img_rgb_3D = np.random.rand(16, 16, 16, 3) * 255
        # Create classification labels
        self.labels_ohe = np.zeros((5, 4), dtype=np.uint8)
        for i in range(0, 5):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1
        # Create xai heatmap
        self.heatmap = np.random.rand(16, 16)
        # Create RGB Data Generator
        self.datagen = DataGenerator(self.sampleList_rgb_2D,
                                     self.tmp_data.name,
                                     labels=self.labels_ohe,
                                     resize=(16, 16),
                                     grayscale=False, batch_size=1)

    #-------------------------------------------------#
    #             Callbacks: CSV2history              #
    #-------------------------------------------------#
    def test_Callbacks_csv2history(self):
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".model")
        path_csv = os.path.join(self.tmp_data.name, "testing.csv")
        csvlog = CSVLogger(path_csv)

        model = NeuralNetwork(n_labels=4, channels=3, input_shape=(16,16))
        hist_returned = model.train(training_generator=self.datagen,
                                    validation_generator=self.datagen,
                                    epochs=3, callbacks=[csvlog])

        hist_loaded = csv_to_history(path_csv)
        del hist_loaded["epoch"]

        for key in hist_returned:
            self.assertTrue(key in hist_returned and key in hist_loaded)
            self.assertTrue(len(hist_returned[key]) == len(hist_loaded[key]))

    #-------------------------------------------------#
    #                Visualizer: Image                #
    #-------------------------------------------------#
    def test_Visualizer_Image_grayscale(self):
        # PNG
        path_out = os.path.join(self.tmp_data.name, "viz.image.2D.gray.png")
        visualize_image(self.img_gray_2D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # JPG
        path_out = os.path.join(self.tmp_data.name, "viz.image.2D.gray.jpg")
        visualize_image(self.img_gray_2D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # TIF
        path_out = os.path.join(self.tmp_data.name, "viz.image.2D.gray.tif")
        visualize_image(self.img_gray_2D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))

    def test_Visualizer_Image_rgb(self):
        # PNG
        path_out = os.path.join(self.tmp_data.name, "viz.image.rgb.png")
        visualize_image(self.img_rgb_2D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # JPG
        path_out = os.path.join(self.tmp_data.name, "viz.image.rgb.jpg")
        visualize_image(self.img_rgb_2D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # TIF
        path_out = os.path.join(self.tmp_data.name, "viz.image.rgb.tif")
        visualize_image(self.img_rgb_2D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))

    #-------------------------------------------------#
    #                Visualizer: Volume               #
    #-------------------------------------------------#
    def test_Visualizer_Volume_grayscale(self):
        # NumPy
        path_out = os.path.join(self.tmp_data.name, "viz.volume.gray.npy")
        visualize_volume(self.img_gray_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # MHA
        path_out = os.path.join(self.tmp_data.name, "viz.volume.gray.mha")
        visualize_volume(self.img_gray_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # NIfTI
        path_out = os.path.join(self.tmp_data.name, "viz.image.gray.nii")
        visualize_volume(self.img_gray_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # GIF
        path_out = os.path.join(self.tmp_data.name, "viz.image.gray.gif")
        visualize_volume(self.img_gray_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))

    def test_Visualizer_Volume_HU(self):
        # NumPy
        path_out = os.path.join(self.tmp_data.name, "viz.volume.hu.npy")
        visualize_volume(self.img_hu_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # MHA
        path_out = os.path.join(self.tmp_data.name, "viz.volume.hu.mha")
        visualize_volume(self.img_hu_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # NIfTI
        path_out = os.path.join(self.tmp_data.name, "viz.image.hu.nii")
        visualize_volume(self.img_hu_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # GIF
        path_out = os.path.join(self.tmp_data.name, "viz.image.hu.gif")
        visualize_volume(self.img_hu_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))

    def test_Visualizer_Volume_rgb(self):
        # NumPy
        path_out = os.path.join(self.tmp_data.name, "viz.volume.rgb.npy")
        visualize_volume(self.img_rgb_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # MHA
        path_out = os.path.join(self.tmp_data.name, "viz.volume.rgb.mha")
        visualize_volume(self.img_rgb_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # NIfTI
        path_out = os.path.join(self.tmp_data.name, "viz.image.rgb.nii")
        visualize_volume(self.img_rgb_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))
        # GIF
        path_out = os.path.join(self.tmp_data.name, "viz.image.rgb.gif")
        visualize_volume(self.img_rgb_3D, out_path=path_out)
        self.assertTrue(os.path.exists(path_out))

    #-------------------------------------------------#
    #             Visualizer: XAI Heatmap             #
    #-------------------------------------------------#
    def test_Visualizer_Heatmap_2D_grayscale(self):
        # Default
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.2D.gray.png")
        visualize_heatmap(self.img_gray_2D, self.heatmap, 
                          overlay=True, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))
        os.remove(path_xai)
        self.assertTrue(not os.path.exists(path_xai))
        # With Overlay
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.2D.gray.png")
        visualize_heatmap(self.img_gray_2D, self.heatmap, 
                          overlay=False, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))

    def test_Visualizer_Heatmap_2D_rgb(self):
        # Default
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.2D.rgb.png")
        visualize_heatmap(self.img_rgb_2D, self.heatmap, 
                          overlay=True, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))
        os.remove(path_xai)
        self.assertTrue(not os.path.exists(path_xai))
        # With Overlay
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.2D.rgb.png")
        visualize_heatmap(self.img_rgb_2D, self.heatmap, 
                          overlay=False, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))

    def test_Visualizer_Heatmap_3D_gray(self):
        # Default
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.3D.gray.npy")
        visualize_heatmap(self.img_gray_3D, self.heatmap, 
                          overlay=True, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))
        os.remove(path_xai)
        self.assertTrue(not os.path.exists(path_xai))
        # With Overlay
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.3D.gray.npy")
        visualize_heatmap(self.img_gray_3D, self.heatmap, 
                          overlay=False, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))

    def test_Visualizer_Heatmap_3D_rgb(self):
        # Default
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.3D.rgb.npy")
        visualize_heatmap(self.img_rgb_3D, self.heatmap, 
                          overlay=True, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))
        os.remove(path_xai)
        self.assertTrue(not os.path.exists(path_xai))
        # With Overlay
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.3D.rgb.npy")
        visualize_heatmap(self.img_rgb_3D, self.heatmap, 
                          overlay=False, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))

    def test_Visualizer_Heatmap_3D_hu(self):
        # Default
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.3D.hu.npy")
        visualize_heatmap(self.img_hu_3D, self.heatmap, 
                          overlay=True, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))
        os.remove(path_xai)
        self.assertTrue(not os.path.exists(path_xai))
        # With Overlay
        path_xai = os.path.join(self.tmp_data.name, "viz.heatmap.3D.hu.npy")
        visualize_heatmap(self.img_hu_3D, self.heatmap, 
                          overlay=False, out_path=path_xai)
        self.assertTrue(os.path.exists(path_xai))


