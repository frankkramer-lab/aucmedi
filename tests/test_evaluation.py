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
import pandas as pd
import random
import tempfile
from PIL import Image
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
        self.labels_ohe = np.zeros((50, 4), dtype=np.uint8)
        for i in range(0, 50):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1
        # Create predictions
        for i in range(0, 50):
            self.preds = np.random.rand(50, 4)
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

        # Create imaging data indices
        self.sample_list = []
        for i in range(0, 50):
            index = "image.sample_" + str(i) + ".RGB.png"
            self.sample_list.append(index)

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

    #-------------------------------------------------#
    #          Evaluation - Plot Performance          #
    #-------------------------------------------------#
    def test_evaluate_performance_minimal(self):
        metrics = evaluate_performance(self.preds, self.labels_ohe,
                                       out_path=self.tmp_plot.name,
                                       multi_label=False, class_names=None,
                                       store_csv=False,
                                       plot_barplot=False,
                                       plot_confusion_matrix=False,
                                       plot_roc_curve=False)
        self.assertTrue(isinstance(metrics, pd.DataFrame))
        self.assertTrue(np.array_equal(metrics.shape, (52, 3)))
        self.assertTrue(np.array_equal(metrics.columns.values,
                                      ["metric", "score", "class"]))

    def test_evaluate_performance_classnames(self):
        metrics = evaluate_performance(self.preds, self.labels_ohe,
                                       out_path=self.tmp_plot.name,
                                       multi_label=False,
                                       class_names=["A", "B", "C", "D"],
                                       store_csv=False,
                                       plot_barplot=False,
                                       plot_confusion_matrix=False,
                                       plot_roc_curve=False)
        classes_unique = np.unique(metrics["class"].to_numpy())
        self.assertTrue(np.array_equal(classes_unique, ["A", "B", "C", "D"]))

    def test_evaluate_performance_barplot(self):
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=False,
                             plot_barplot=True,
                             plot_confusion_matrix=False,
                             plot_roc_curve=False)
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.performance.barplot.png")
        self.assertTrue(os.path.exists(path_plot))
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=False,
                             plot_barplot=True,
                             plot_confusion_matrix=False,
                             plot_roc_curve=False,
                             suffix="test")
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.performance.barplot.test.png")
        self.assertTrue(os.path.exists(path_plot))

    def test_evaluate_performance_confusionmatrix(self):
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=False,
                             plot_barplot=False,
                             plot_confusion_matrix=True,
                             plot_roc_curve=False)
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.performance.confusion_matrix.png")
        self.assertTrue(os.path.exists(path_plot))
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=False,
                             plot_barplot=False,
                             plot_confusion_matrix=True,
                             plot_roc_curve=False,
                             suffix="test")
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.performance.confusion_matrix.test.png")
        self.assertTrue(os.path.exists(path_plot))

    def test_evaluate_performance_roc(self):
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=False,
                             plot_barplot=False,
                             plot_confusion_matrix=False,
                             plot_roc_curve=True)
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.performance.roc.png")
        self.assertTrue(os.path.exists(path_plot))
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=False,
                             plot_barplot=False,
                             plot_confusion_matrix=False,
                             plot_roc_curve=True,
                             suffix="test")
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.performance.roc.test.png")
        self.assertTrue(os.path.exists(path_plot))

    def test_evaluate_performance_csv(self):
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=True,
                             plot_barplot=False,
                             plot_confusion_matrix=False,
                             plot_roc_curve=False)
        path_csv = os.path.join(self.tmp_plot.name,
                                "metrics.performance.csv")
        self.assertTrue(os.path.exists(path_csv))
        evaluate_performance(self.preds, self.labels_ohe,
                             out_path=self.tmp_plot.name,
                             multi_label=False,
                             store_csv=True,
                             plot_barplot=False,
                             plot_confusion_matrix=False,
                             plot_roc_curve=False,
                             suffix="test")
        path_csv = os.path.join(self.tmp_plot.name,
                                "metrics.performance.test.csv")
        self.assertTrue(os.path.exists(path_csv))

    #-------------------------------------------------#
    #          Evaluation - Plot Comparison           #
    #-------------------------------------------------#
    def test_evaluate_comparison_minimal(self):
        pred_list = [self.preds,
                     np.flip(self.preds, axis=0),
                     np.flip(self.preds, axis=1)]

        df_merged, df_gain = evaluate_comparison(pred_list, self.labels_ohe,
                                                 out_path=self.tmp_plot.name)

        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.comparison.beside.png")
        self.assertTrue(os.path.exists(path_plot))
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.comparison.gain.png")
        self.assertTrue(os.path.exists(path_plot))

        self.assertTrue(np.array_equal(df_merged.shape, (156, 4)))
        self.assertTrue(np.array_equal(df_gain.shape, (108, 4)))

    def test_evaluate_comparison_naming(self):
        pred_list = [self.preds,
                     np.flip(self.preds, axis=0),
                     np.flip(self.preds, axis=1)]

        df_merged, df_gain = evaluate_comparison(pred_list, self.labels_ohe,
                                                 out_path=self.tmp_plot.name,
                                                 class_names=["A", "B", "C", "D"],
                                                 model_names=["x", "y", "z"],)

        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.comparison.beside.png")
        self.assertTrue(os.path.exists(path_plot))
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.comparison.gain.png")
        self.assertTrue(os.path.exists(path_plot))

        classes_unique = np.unique(df_merged["class"].to_numpy())
        self.assertTrue(np.array_equal(classes_unique, ["A", "B", "C", "D"]))
        models_unique = np.unique(df_merged["model"].to_numpy())
        self.assertTrue(np.array_equal(models_unique, ["x", "y", "z"]))

    def test_evaluate_comparison_macroaveraged(self):
        pred_list = [self.preds,
                     np.flip(self.preds, axis=0),
                     np.flip(self.preds, axis=1)]

        df_merged, df_gain = evaluate_comparison(pred_list, self.labels_ohe,
                                                 out_path=self.tmp_plot.name,
                                                 macro_average_classes=True)

        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.comparison.beside.png")
        self.assertTrue(os.path.exists(path_plot))
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.comparison.gain.png")
        self.assertTrue(os.path.exists(path_plot))

        self.assertTrue(np.array_equal(df_merged.shape, (39, 3)))
        self.assertTrue(np.array_equal(df_gain.shape, (27, 3)))

    #-------------------------------------------------#
    #          Evaluation - Dataset Analysis          #
    #-------------------------------------------------#
    def test_evaluate_dataset(self):
        res = evaluate_dataset(self.sample_list, self.labels_ohe,
                               out_path=self.tmp_plot.name,
                               class_names=["A", "B", "C", "D"])
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.dataset.heatmap.png")
        self.assertFalse(os.path.exists(path_plot))
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.dataset.barplot.png")
        self.assertFalse(os.path.exists(path_plot))
        self.assertTrue(isinstance(res, pd.DataFrame))
        self.assertTrue(self.labels_ohe.shape[1] == res.shape[0])

    def test_evaluate_dataset_barplot(self):
        res = evaluate_dataset(self.sample_list, self.labels_ohe,
                               out_path=self.tmp_plot.name,
                               class_names=["A", "B", "C", "D"],
                               plot_barplot=True)
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.dataset.heatmap.png")
        self.assertFalse(os.path.exists(path_plot))
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.dataset.barplot.png")
        self.assertTrue(os.path.exists(path_plot))
        os.remove(path_plot)
        self.assertTrue(isinstance(res, pd.DataFrame))
        self.assertTrue(self.labels_ohe.shape[1] == res.shape[0])

    def test_evaluate_dataset_heatmap(self):
        res = evaluate_dataset(self.sample_list, self.labels_ohe,
                               out_path=self.tmp_plot.name,
                               class_names=["A", "B", "C", "D"],
                               plot_heatmap=True)
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.dataset.heatmap.png")
        self.assertTrue(os.path.exists(path_plot))
        os.remove(path_plot)
        path_plot = os.path.join(self.tmp_plot.name,
                                 "plot.dataset.barplot.png")
        self.assertFalse(os.path.exists(path_plot))
        self.assertTrue(isinstance(res, pd.DataFrame))
        self.assertTrue(self.labels_ohe.shape[1] == res.shape[0])
