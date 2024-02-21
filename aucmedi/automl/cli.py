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
#                    Documentation                    #
#-----------------------------------------------------#
""" Argparse for the AutoML Command Line Interface of [aucmedi.automl.main][aucmedi.automl.main].

The parameters are summarized in the docs: [Documentation - AutoML - Parameters](../../../automl/parameters/)
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import argparse
import pkg_resources
import sys

#-----------------------------------------------------#
#                    CLI - General                    #
#-----------------------------------------------------#
def cli_core():
    """ Internal function for Command-Line-Interface (CLI) setup. """
    # Set description for cli core
    desc = """ AutoML command-line interface for AUCMEDI: a framework for
               Automated Classification of Medical Images """
    # Setup ArgumentParser interface
    parser = argparse.ArgumentParser(prog="aucmedi", description=desc)
    # Add optional core arguments
    version = pkg_resources.require("aucmedi")[0].version
    parser.add_argument("-v", "--version", action="version",
                        version='%(prog)s_v' + version)

    # Add subparser interface
    subparsers = parser.add_subparsers(title="Application Modes",
                                       dest="hub")
    # Return parsers
    return parser, subparsers

# #-----------------------------------------------------#
# #                     CLI - YAML                      #
# #-----------------------------------------------------#
# def cli_yaml(subparsers):
#     # Set description for cli training
#     desc = """ YAML interface for reading configurations from a file """
#     # Setup SubParser
#     parser_yaml = subparsers.add_parser("yaml", help=desc, add_help=False)
#
#     # Add required configuration arguments
#     ra = parser_yaml.add_argument_group("required arguments")
#     ra.add_argument("-i", "--input",
#                     type=str,
#                     required=True,
#                     help="Path to a YAML file with AUCMEDI AutoML " + \
#                          "configurations")
#
#     # Add optional configuration arguments
#     oa = parser_yaml.add_argument_group("optional arguments")
#     oa.add_argument("-h",
#                     "--help",
#                     action="help",
#                     help="show this help message and exit")

#-----------------------------------------------------#
#                    CLI - Training                   #
#-----------------------------------------------------#
def cli_training(subparsers):
    """ Parameter overview for the training process.

    | Category      | Argument               | Type       | Default        | Description |
    | :------------ | :--------------------- | :--------- | :------------- | :---------- |
    | I/O           | `--path_imagedir`      | str        | `training`     | Path to the directory containing the images. |
    | I/O           | `--path_modeldir`      | str        | `model`        | Path to the output directory in which fitted models and metadata are stored. |
    | I/O           | `--path_gt`            | str        | `None`         | Path to the index/class annotation file if required. (only for 'csv' interface). |
    | I/O           | `--ohe`                | bool       | `False`        | Boolean option whether annotation data is sparse categorical or one-hot encoded. |
    | Configuration | `--analysis`           | str        | `standard`     | Analysis mode for the AutoML training. Options: `["minimal", "standard", "advanced"]`. |
    | Configuration | `--three_dim`          | bool       | `False`        | Boolean, whether data is 2D or 3D. |
    | Configuration | `--shape_3D`           | str        | `128x128x128`  | Desired input shape of 3D volume for architecture (will be cropped into, format: `1x2x3`). |
    | Configuration | `--epochs`             | int        | `500`          | Number of epochs. A single epoch is defined as one iteration through the complete data set. |
    | Configuration | `--batch_size`         | int        | `24`           | Number of samples inside a single batch. |
    | Configuration | `--workers`            | int        | `1`            | Number of workers/threads which preprocess batches during runtime. |
    | Configuration | `--metalearner`        | str        | `mean`         | Key for Metalearner or Aggregate function. |
    | Configuration | `--architecture`       | str        | `DenseNet121`  | Key of single or multiple Architectures (only supported for 'analysis=advanced', format: 'KEY' or 'KEY,KEY,KEY). |
    | Other         | `--help`               | bool       | `False`        | show this help message and exit. |

    ??? info "List of Architectures"
        AUCMEDI provides a large library of state-of-the-art and ready-to-use architectures.

        - 2D Architectures: [aucmedi.neural_network.architectures.image][]
        - 3D Architectures: [aucmedi.neural_network.architectures.volume][]

    ??? info "List of Metalearner"
        - Homogeneous pooling functions: [Aggregate][aucmedi.ensemble.aggregate]
        - Heterogeneous pooling functions: [Metalearner][aucmedi.ensemble.metalearner]
    """
    # Set description for cli training
    desc = """ Pipeline hub for Training via AUCMEDI AutoML """
    # Setup SubParser
    parser_train = subparsers.add_parser("training", help=desc, add_help=False)

    # Add IO arguments
    od = parser_train.add_argument_group("Arguments - I/O")
    od.add_argument("--path_imagedir",
                    type=str,
                    required=False,
                    default="training",
                    help="Path to the directory containing the images " + \
                         "(default: '%(default)s')",
                    )
    od.add_argument("--path_modeldir",
                    type=str,
                    required=False,
                    default="model",
                    help="Path to the output directory in which fitted " + \
                         "models and metadata are stored " + \
                         "(default: '%(default)s')",
                    )
    od.add_argument("--path_gt",
                    type=str,
                    required=False,
                    help="Path to the index/class annotation CSV file " + \
                         "(only required for defining the ground truth via " + \
                         "'csv' instead of 'directory' interface)",
                    )

    od.add_argument("--ohe",
                    action="store_true",
                    required=False,
                    default=False,
                    help="Boolean option whether annotation data is sparse " + \
                         "categorical or one-hot encoded " + \
                         "(only required for interface 'csv' and multi-" + \
                         "label data, " + \
                         "default: '%(default)s')",
                    )

    # Add configuration arguments
    oc = parser_train.add_argument_group("Arguments - Configuration")
    oc.add_argument("--analysis",
                    type=str,
                    required=False,
                    default="standard",
                    choices=["minimal", "standard", "advanced"],
                    help="Analysis mode for the AutoML training " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--three_dim",
                    action="store_true",
                    required=False,
                    default=False,
                    help="Boolean, whether imaging data is 2D or 3D " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--shape_3D",
                    type=str,
                    required=False,
                    default="128x128x128",
                    help="Desired input shape of 3D volume for architecture "+ \
                         "(will be cropped into, " + \
                         "format: '1x2x3', " + \
                         "default: '%(default)s')",
                    )
    oc.add_argument("--epochs",
                    type=int,
                    required=False,
                    default=500,
                    help="Number of epochs. A single epoch is defined as " + \
                         "one iteration through the complete data set " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--batch_size",
                    type=int,
                    required=False,
                    default=24,
                    help="Number of samples inside a single batch " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--workers",
                    type=int,
                    required=False,
                    default=1,
                    help="Number of workers/threads which preprocess " + \
                         "batches during runtime " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--metalearner",
                    type=str,
                    required=False,
                    default="mean",
                    help="Key for Metalearner or Aggregate function "+ \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--architecture",
                    type=str,
                    required=False,
                    default="DenseNet121",
                    help="Key of single or multiple Architectures " + \
                         "(multiple Architectures are only supported for " + \
                         "'analysis=advanced', " + \
                         "format: 'KEY' or 'KEY,KEY,KEY', " + \
                         "default: '%(default)s')",
                    )

    # Add other arguments
    oo = parser_train.add_argument_group("Arguments - Other")
    oo.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")

#-----------------------------------------------------#
#                   CLI - Prediction                  #
#-----------------------------------------------------#
def cli_prediction(subparsers):
    """ Parameter overview for the prediction process.

    | Category      | Argument               | Type       | Default        | Description |
    | :------------ | :--------------------- | :--------- | :------------- | :---------- |
    | I/O           | `--path_imagedir`      | str        | `test`         | Path to the directory containing the images. |
    | I/O           | `--path_modeldir`      | str        | `model`        | Path to the output directory in which fitted models and metadata are stored. |
    | I/O           | `--path_pred`          | str        | `preds.csv`    | Path to the output file in which predicted csv file should be stored. |
    | Configuration | `--xai_method`         | str        | `None`         | Key for XAI method.  |
    | Configuration | `--xai_directory`      | str        | `xai`          | Path to the output directory in which predicted image xai heatmaps should be stored. |
    | Configuration | `--batch_size`         | int        | `24`           | Number of samples inside a single batch. |
    | Configuration | `--workers`            | int        | `1`            | Number of workers/threads which preprocess batches during runtime. |
    | Other         | `--help`               | bool       | `False`        | show this help message and exit. |

    ??? info "List of XAI Methods"
        AUCMEDI provides a large library of state-of-the-art and ready-to-use XAI methods:
        [aucmedi.xai.methods][]
    """
    # Set description for cli prediction
    desc = """ Pipeline hub for Inference via AUCMEDI AutoML """
    # Setup SubParser
    parser_predict = subparsers.add_parser("prediction",
                                           help=desc,
                                           add_help=False)

    # Add IO arguments
    od = parser_predict.add_argument_group("Arguments - I/O")
    od.add_argument("--path_imagedir",
                    type=str,
                    required=False,
                    default="test",
                    help="Path to the directory containing the images " + \
                         "(default: '%(default)s')",
                    )
    od.add_argument("--path_modeldir",
                    type=str,
                    required=False,
                    default="model",
                    help="Path to the model directory in which fitted " + \
                         "model weights and metadata are stored " + \
                         "(default: '%(default)s')",
                    )
    od.add_argument("--path_pred",
                    type=str,
                    required=False,
                    default="preds.csv",
                    help="Path to the output file in which predicted csv " + \
                         "file should be stored " + \
                         "(default: '%(default)s')",
                    )

    # Add configuration arguments
    oc = parser_predict.add_argument_group("Arguments - Configuration")
    oc.add_argument("--xai_method",
                    type=str,
                    required=False,
                    help="Key for XAI method " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--xai_directory",
                    type=str,
                    required=False,
                    default="xai",
                    help="Path to the output directory in which predicted " + \
                         "image xai heatmaps should be stored " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--batch_size",
                    type=int,
                    required=False,
                    default=12,
                    help="Number of samples inside a single batch " + \
                         "(default: '%(default)s')",
                    )
    oc.add_argument("--workers",
                    type=int,
                    required=False,
                    default=1,
                    help="Number of workers/threads which preprocess " + \
                         "batches during runtime " + \
                         "(default: '%(default)s')",
                    )

    # Add other arguments
    oo = parser_predict.add_argument_group("Arguments - Other")
    oo.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")

#-----------------------------------------------------#
#                   CLI - Evaluation                  #
#-----------------------------------------------------#
def cli_evaluation(subparsers):
    """ Parameter overview for the evaluation process.

    | Category      | Argument               | Type       | Default        | Description |
    | :------------ | :--------------------- | :--------- | :------------- | :---------- |
    | I/O           | `--path_imagedir`      | str        | `training`     | Path to the directory containing the ground truth images. |
    | I/O           | `--path_gt`            | str        | `None`         | Path to the index/class annotation CSV file (only required for defining the ground truth via 'csv' instead of 'directory' interface). |
    | I/O           | `--ohe`                | bool       | `False`        | Boolean option whether annotation data is sparse categorical or one-hot encoded. |
    | I/O           | `--path_pred`          | str        | `preds.csv`    | Path to the input file in which predicted csv file is stored. |
    | I/O           | `--path_evaldir`       | str        | `evaluation`   | Path to the directory in which evaluation figures and tables should be stored. |
    | Other         | `--help`               | bool       | `False`        | show this help message and exit. |
    """
    # Set description for cli evaluation
    desc = """ Pipeline hub for Evaluation via AUCMEDI AutoML """
    # Setup SubParser
    parser_evaluate = subparsers.add_parser("evaluation",
                                            help=desc,
                                            add_help=False)

    # Add IO arguments
    od = parser_evaluate.add_argument_group("Arguments - I/O")
    od.add_argument("--path_imagedir",
                    type=str,
                    required=False,
                    default="training",
                    help="Path to the directory containing the ground truth" + \
                         " images " + \
                         "(default: '%(default)s')",
                    )
    od.add_argument("--path_gt",
                    type=str,
                    required=False,
                    help="Path to the index/class annotation CSV file " + \
                         "(only required for defining the ground truth via " + \
                         "'csv' instead of 'directory' interface)",
                    )
    od.add_argument("--ohe",
                    action="store_true",
                    required=False,
                    default=False,
                    help="Boolean option whether annotation data is sparse " + \
                         "categorical or one-hot encoded " + \
                         "(only required for interface 'csv' and multi-" + \
                         "label data, " + \
                         "default: '%(default)s')",
                    )
    od.add_argument("--path_pred",
                    type=str,
                    required=False,
                    default="preds.csv",
                    help="Path to the output file in which predicted csv " + \
                         "file are stored " + \
                         "(default: '%(default)s')",
                    )
    od.add_argument("--path_evaldir",
                    type=str,
                    required=False,
                    default="evaluation",
                    help="Path to the directory in which evaluation " + \
                         "figures and tables should be stored " + \
                         "(default: '%(default)s')",
                    )

    # Add other arguments
    oo = parser_evaluate.add_argument_group("Arguments - Other")
    oo.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")
