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
import tempfile
import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
import random
import pandas as pd
#Internal libraries
from aucmedi.automl.block_train import block_train

#-----------------------------------------------------#
#           Unittest: AutoML Training Block           #
#-----------------------------------------------------#
class AutoML_block_train(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data2D = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                      suffix=".data2D")
        self.tmp_data3D = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data3D")

        # Create RGB data
        for i in range(0, 25):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "sample_" + str(i) + ".png"
            path_sampleRGB = os.path.join(self.tmp_data2D.name, index)
            imgRGB_pillow.save(path_sampleRGB)

        # Create nii dataset
        for i in range(0, 25):
            index = "sample_" + str(i) + ".nii"
            path_sample = os.path.join(self.tmp_data3D.name, index)
            img_hu = np.float32(np.random.rand(16, 16, 16, 1) * 1500 - 500)
            image_sitk = sitk.GetImageFromArray(img_hu)
            image_sitk.SetSpacing([1.75,1.25,0.75])
            sitk.WriteImage(image_sitk, path_sample)

        # Create multi-class classification labels
        data = {}
        for i in range(0, 25):
            data["sample_" + str(i)] = np.random.randint(4)
        self.tmp_csv = tempfile.NamedTemporaryFile(mode="w",
                                                   prefix="tmp.aucmedi.",
                                                   suffix=".csv")
        df = pd.DataFrame.from_dict(data, orient="index", columns=["CLASS"])
        df.index.name = "SAMPLE"
        df.to_csv(self.tmp_csv.name, index=True, header=True)

        # Create multi-label classification labels
        data = {}
        for i in range(0, 25):
            labels_ohe = [0, 0, 0, 0]
            class_index = np.random.randint(0, 4)
            labels_ohe[class_index] = 1
            class_index = np.random.randint(0, 4)
            labels_ohe[class_index] = 1
            data["sample_" + str(i)] = labels_ohe
        self.tmp_csv_ohe = tempfile.NamedTemporaryFile(mode="w",
                                                       prefix="tmp.aucmedi.",
                                                       suffix=".csv")
        df = pd.DataFrame.from_dict(data, orient="index",
                                    columns=["a", "b", "c", "d"])
        df.index.name = "SAMPLE"
        df.to_csv(self.tmp_csv_ohe.name, index=True, header=True)

    #-------------------------------------------------#
    #                Analysis: Minimal                #
    #-------------------------------------------------#
    def test_minimal(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": output_dir.name,
            "analysis": "minimal",
            "ohe": False,
            "three_dim": False,
            "shape_3D": (128,128,128),
            "epochs": 8,
            "batch_size": 4,
            "workers": 1,
            "metalearner": "logistic_regression",
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.last.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "logs.training.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "meta.training.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "plot.fitting_course.png")))

    def test_minimal_multilabel(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv_ohe.name,
            "path_modeldir": output_dir.name,
            "analysis": "minimal",
            "ohe": True,
            "three_dim": False,
            "shape_3D": (128,128,128),
            "epochs": 8,
            "batch_size": 4,
            "workers": 1,
            "metalearner": "mean",
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.last.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "logs.training.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "meta.training.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "plot.fitting_course.png")))

    def test_minimal_3D(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data3D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": output_dir.name,
            "analysis": "minimal",
            "ohe": False,
            "three_dim": True,
            "shape_3D": (16, 16, 16),
            "epochs": 8,
            "batch_size": 1,
            "workers": 1,
            "metalearner": "logistic_regression",
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.last.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "logs.training.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "meta.training.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "plot.fitting_course.png")))

    #-------------------------------------------------#
    #                Analysis: Standard               #
    #-------------------------------------------------#
    def test_standard(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": output_dir.name,
            "analysis": "standard",
            "ohe": False,
            "three_dim": False,
            "epochs": 8,
            "batch_size": 4,
            "workers": 1,
            "metalearner": "logistic_regression",
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.last.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.best_loss.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "logs.training.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "meta.training.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "plot.fitting_course.png")))

    def test_standard_multilabel(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv_ohe.name,
            "path_modeldir": output_dir.name,
            "analysis": "standard",
            "ohe": True,
            "three_dim": False,
            "epochs": 8,
            "batch_size": 4,
            "workers": 1,
            "metalearner": "mean",
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.last.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.best_loss.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "logs.training.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "meta.training.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "plot.fitting_course.png")))

    def test_standard_3D(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data3D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": output_dir.name,
            "analysis": "standard",
            "ohe": False,
            "three_dim": True,
            "shape_3D": (16, 16, 16),
            "epochs": 8,
            "batch_size": 1,
            "workers": 1,
            "metalearner": "logistic_regression",
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.last.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "model.best_loss.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "logs.training.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "meta.training.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir.name, "plot.fitting_course.png")))

    #-------------------------------------------------#
    #               Analysis: Composite               #
    #-------------------------------------------------#
    def test_composite(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": output_dir.name,
            "analysis": "advanced",
            "ohe": False,
            "three_dim": False,
            "epochs": 8,
            "batch_size": 1,
            "workers": 1,
            "metalearner": "logistic_regression",
            "architecture": ["Vanilla", "Vanilla"]
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(len(os.listdir(output_dir.name))==7)

    def test_composite_multilabel(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv_ohe.name,
            "path_modeldir": output_dir.name,
            "analysis": "advanced",
            "ohe": True,
            "three_dim": False,
            "epochs": 8,
            "batch_size": 1,
            "workers": 1,
            "metalearner": "mean",
            "architecture": ["Vanilla", "Vanilla"]
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(len(os.listdir(output_dir.name))==6)

    def test_composite_3D(self):
        # Initialize temporary directory
        output_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                 suffix=".output")
        # Define config
        config = {
            "interface": "csv",
            "path_imagedir": self.tmp_data3D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": output_dir.name,
            "analysis": "advanced",
            "ohe": False,
            "three_dim": True,
            "shape_3D": (16, 16, 16),
            "epochs": 8,
            "batch_size": 1,
            "workers": 1,
            "metalearner": "logistic_regression",
            "architecture": ["Vanilla", "Vanilla"]
        }
        # Run AutoML training block
        block_train(config)

        self.assertTrue(len(os.listdir(output_dir.name))==7)
