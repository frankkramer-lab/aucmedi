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
import random
import tempfile
import os
#Internal libraries
from aucmedi import *
from aucmedi.evaluation import *

#-----------------------------------------------------#
#                 Unittest: Evaluation                #
#-----------------------------------------------------#
class EvaluationTEST(unittest.TestCase):
    # Create random imaging data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_plot = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".plot")

        # Create classification labels
        self.labels_ohe = np.zeros((1, 4), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create artificial history data - basic
        self.hist_basic = {"loss": []}
        for i in range(0, 150):
            self.hist_basic["loss"].append(random.uniform(0, 1))
        # Create artificial history data - standard
        self.hist_standard = {"loss": [], "val_loss": []}
        for i in range(0, 150):
            self.hist_standard["loss"].append(random.uniform(0, 1))
            self.hist_standard["val_loss"].append(random.uniform(0, 1))
        # Create artificial history data - advanced
        self.hist_advanced = {"loss": [], "val_loss": [],
                              "accuracy": [], "val_accuracy": []}
        for i in range(0, 150):
            self.hist_advanced["loss"].append(random.uniform(0, 1))
            self.hist_advanced["val_loss"].append(random.uniform(0, 1))
            self.hist_advanced["accuracy"].append(random.uniform(0, 1))
            self.hist_advanced["val_accuracy"].append(random.uniform(0, 1))
        # Create artificial history data - bagging
        self.hist_bagging = {}
        for cv in range(0, 3):
            metrics = ["loss", "val_loss", "accuracy", "val_accuracy"]
            for m in metrics:
                self.hist_bagging["cv_" + str(cv) + "." + m] = []
                for i in range(0, 150):
                    self.hist_bagging["cv_" + str(cv) + "." + m].append(
                         random.uniform(0, 1))
        # Create artificial history data - stacking
        self.hist_stacking = {}
        for nn in range(0, 3):
            metrics = ["loss", "val_loss", "accuracy", "val_accuracy"]
            for m in metrics:
                self.hist_stacking["nn_" + str(nn) + "." + m] = []
                for i in range(0, 150):
                    self.hist_stacking["nn_" + str(nn) + "." + m].append(
                         random.uniform(0, 1))

    #-------------------------------------------------#
    #            Evaluation - Plot Fitting            #
    #-------------------------------------------------#
    def test_evaluate_fitting_basic(self):
        evaluate_fitting(self.hist_basic, out_path=self.tmp_plot.name,
                         monitor=["loss"], suffix="basic")
        self.assertTrue(os.path.exists(os.path.join(self.tmp_plot.name,
                                       "plot.fitting_course.basic.png")))

    def test_evaluate_fitting_standard(self):
        evaluate_fitting(self.hist_standard, out_path=self.tmp_plot.name,
                        monitor=["loss"], suffix="standard")
        self.assertTrue(os.path.exists(os.path.join(self.tmp_plot.name,
                                      "plot.fitting_course.standard.png")))

    def test_evaluate_fitting_advanced(self):
        evaluate_fitting(self.hist_advanced, out_path=self.tmp_plot.name,
                       monitor=["loss", "accuracy"], suffix="advanced")
        self.assertTrue(os.path.exists(os.path.join(self.tmp_plot.name,
                                     "plot.fitting_course.advanced.png")))

    def test_evaluate_fitting_bagging(self):
        evaluate_fitting(self.hist_bagging, out_path=self.tmp_plot.name,
                      monitor=["loss", "accuracy"], suffix="bagging")
        self.assertTrue(os.path.exists(os.path.join(self.tmp_plot.name,
                                    "plot.fitting_course.bagging.png")))

    def test_evaluate_fitting_stacking(self):
        evaluate_fitting(self.hist_stacking, out_path=self.tmp_plot.name,
                         monitor=["loss"], suffix="stacking")
        self.assertTrue(os.path.exists(os.path.join(self.tmp_plot.name,
                                       "plot.fitting_course.stacking.png")))
