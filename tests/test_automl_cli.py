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
import numpy as np
from unittest.mock import patch
import sys
from shutil import which
#Internal libraries
from aucmedi.automl.main import main
from aucmedi.automl.cli import *
from aucmedi.automl import parse_cli

#-----------------------------------------------------#
#                Unittest: AutoML CLI                 #
#-----------------------------------------------------#
class AutoML_CLI(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create temporary directory-based imaging data
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        for i in range(0, 5):
            os.mkdir(os.path.join(self.tmp_data.name, "class_" + str(i)))
        # Fill subdirectories with images
        for i in range(0, 25):
            img = np.random.rand(16, 16, 3) * 255
            img_pillow = Image.fromarray(img.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            label_dir = "class_" + str((i % 5))
            path_sample = os.path.join(self.tmp_data.name, label_dir, index)
            img_pillow.save(path_sample)
        # Create temporary model directory
        self.tmp_model = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                     suffix=".model")

    #-------------------------------------------------#
    #                  CLI Hub: Core                  #
    #-------------------------------------------------#
    def test_core_empty(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        args = ["aucmedi"]
        with patch.object(sys, "argv", args):
            with self.assertRaises(SystemExit) as se:
                main()
            self.assertEqual(se.exception.code, 1)

    def test_core_version(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        with patch.object(sys, "argv", ["aucmedi", "-v"]):
            with self.assertRaises(SystemExit) as se:
                main()
        with patch.object(sys, "argv", ["aucmedi", "--version"]):
            with self.assertRaises(SystemExit) as se:
                main()

    def test_core_help(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        with patch.object(sys, "argv", ["aucmedi", "-h"]):
            with self.assertRaises(SystemExit) as se:
                main()
        with patch.object(sys, "argv", ["aucmedi", "--help"]):
            with self.assertRaises(SystemExit) as se:
                main()

    #-------------------------------------------------#
    #                CLI Hub: Training                #
    #-------------------------------------------------#
    def test_training_functionality(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        args = ["aucmedi", "training"]
        args_config = ["--path_imagedir", self.tmp_data.name,
                       "--epochs", "1",
                       "--architecture", "Vanilla",
                       "--path_modeldir", self.tmp_model.name]
        with patch.object(sys, "argv", args + args_config):
            main()

    def test_training_args(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        args = ["aucmedi", "training"]
        args_config = ["--path_imagedir", self.tmp_data.name]
        # Build and run CLI functions
        with patch.object(sys, "argv", args + args_config):
            parser, subparsers = cli_core()
            cli_training(subparsers)
            args = parser.parse_args()
            config_cli = parse_cli(args)
        # Define possible config parameters
        config_map = ["path_imagedir",
                      "path_gt",
                      "path_modeldir",
                      "analysis",
                      "ohe",
                      "three_dim",
                      "shape_3D",
                      "epochs",
                      "batch_size",
                      "workers",
                      "metalearner",
                      "architecture",
                     ]
        # Check existence
        for c in config_map:
            self.assertTrue(c in config_cli)

    #-------------------------------------------------#
    #               CLI Hub: Prediction               #
    #-------------------------------------------------#
    def test_prediction_functionality(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        # Training
        args = ["aucmedi", "training"]
        args_config = ["--path_imagedir", self.tmp_data.name,
                       "--epochs", "1",
                       "--architecture", "Vanilla",
                       "--path_modeldir", self.tmp_model.name]
        with patch.object(sys, "argv", args + args_config):
            main()
        # Prediction
        tmp_output = tempfile.NamedTemporaryFile(mode="w",
                                                 prefix="tmp.aucmedi.",
                                                 suffix=".pred.csv")
        args = ["aucmedi", "prediction"]
        args_config = ["--path_imagedir", os.path.join(self.tmp_data.name,
                                                       "class_0"),
                       "--path_modeldir", self.tmp_model.name,
                       "--path_pred", tmp_output.name]
        with patch.object(sys, "argv", args + args_config):
            main()

    def test_prediction_args(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        args = ["aucmedi", "prediction"]
        args_config = ["--path_imagedir", os.path.join(self.tmp_data.name,
                                                         "class_0")]
        # Build and run CLI functions
        with patch.object(sys, "argv", args + args_config):
            parser, subparsers = cli_core()
            cli_prediction(subparsers)
            args = parser.parse_args()
            config_cli = parse_cli(args)
        # Define possible config parameters
        config_map = ["path_imagedir",
                      "path_modeldir",
                      "path_pred",
                      "xai_method",
                      "xai_directory",
                      "batch_size",
                      "workers",
                     ]
        # Check existence
        for c in config_map:
            self.assertTrue(c in config_cli)

    #-------------------------------------------------#
    #               CLI Hub: Evaluation               #
    #-------------------------------------------------#
    def test_evaluation_functionality(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        # Training
        args = ["aucmedi", "training"]
        args_config = ["--path_imagedir", self.tmp_data.name,
                       "--epochs", "1",
                       "--architecture", "Vanilla",
                       "--path_modeldir", self.tmp_model.name]
        with patch.object(sys, "argv", args + args_config):
            main()
        # Prediction
        tmp_output = tempfile.NamedTemporaryFile(mode="w",
                                                 prefix="tmp.aucmedi.",
                                                 suffix=".pred.csv")
        args = ["aucmedi", "prediction"]
        args_config = ["--path_imagedir", os.path.join(self.tmp_data.name,
                                                       "class_0"),
                       "--path_modeldir", self.tmp_model.name,
                       "--path_pred", tmp_output.name]
        with patch.object(sys, "argv", args + args_config):
            main()
        # Evaluation
        tmp_eval = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".eval")
        args = ["aucmedi", "evaluation"]
        args_config = ["--path_imagedir", self.tmp_data.name,
                       "--path_pred", tmp_output.name,
                       "--path_evaldir", tmp_eval.name]
        with patch.object(sys, "argv", args + args_config):
            main()

    def test_evaluation_args(self):
        if which("aucmedi") is None : return    # only check unittesting for build (install via pip)
        args = ["aucmedi", "evaluation"]
        args_config = ["--path_imagedir", self.tmp_data.name]
        # Build and run CLI functions
        with patch.object(sys, "argv", args + args_config):
            parser, subparsers = cli_core()
            cli_evaluation(subparsers)
            args = parser.parse_args()
            config_cli = parse_cli(args)
        # Define possible config parameters
        config_map = ["path_gt",
                      "path_pred",
                      "path_evaldir",
                      "ohe",
                     ]
        # Check existence
        for c in config_map:
            self.assertTrue(c in config_cli)
