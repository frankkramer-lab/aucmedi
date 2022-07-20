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
from aucmedi.automl.block_pred import block_predict
from aucmedi.automl.block_eval import block_evaluate

#-----------------------------------------------------#
#          Unittest: AutoML Evaluation Block          #
#-----------------------------------------------------#
class AutoML_block_evaluate(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data2D = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                      suffix=".data2D")
        # Create RGB data
        for i in range(0, 25):
            img_rgb = np.random.rand(32, 32, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "sample_" + str(i) + ".png"
            path_sampleRGB = os.path.join(self.tmp_data2D.name, index)
            imgRGB_pillow.save(path_sampleRGB)

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

        # Initialize temporary directory
        self.model_dir = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                     suffix=".train")
        # Define config
        config = {
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv.name,
            "path_modeldir": self.model_dir.name,
            "analysis": "minimal",
            "ohe": False,
            "three_dim": False,
            "epochs": 8,
            "batch_size": 4,
            "workers": 1,
            "architecture": "Vanilla"
        }
        # Run AutoML training block
        block_train(config)

        # Define config
        self.pred_path = tempfile.NamedTemporaryFile(mode="w",
                                                     prefix="tmp.aucmedi.",
                                                     suffix=".pred.csv")
        config = {
            "path_imagedir": self.tmp_data2D.name,
            "path_modeldir": self.model_dir.name,
            "path_pred": self.pred_path.name,
            "batch_size": 4,
            "workers": 1,
            "xai_method": None,
            "xai_directory": None,
        }
        # Run AutoML inference block
        block_predict(config)

    #-------------------------------------------------#
    #              Performance Evaluation             #
    #-------------------------------------------------#
    def test_eval_performance(self):
        # Define config
        config = {
            "path_imagedir": self.tmp_data2D.name,
            "path_gt": self.tmp_csv.name,
            "path_pred": self.pred_path.name,
            "path_evaldir": self.model_dir.name,
            "ohe": False,
        }
        # Run evaluation block
        block_evaluate(config)

        self.assertTrue(len(os.listdir(self.model_dir.name)) == 8)
        self.assertTrue(os.path.exists(os.path.join(self.model_dir.name,
                                       "plot.performance.barplot.png")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir.name,
                                      "plot.performance.confusion_matrix.png")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir.name,
                                      "plot.performance.roc.png")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir.name,
                                      "metrics.performance.csv")))
