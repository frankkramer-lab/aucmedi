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
# External libraries
import argparse
import pkg_resources
import sys

#-----------------------------------------------------#
#                    CLI - General                    #
#-----------------------------------------------------#
def cli_core():
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

#-----------------------------------------------------#
#                     CLI - YAML                      #
#-----------------------------------------------------#
def cli_yaml(subparsers):
    # Set description for cli training
    desc = """ YAML interface for reading configurations from a file """
    # Setup SubParser
    parser_yaml = subparsers.add_parser("yaml", help=desc, add_help=False)

    # Add required configuration arguments
    ra = parser_yaml.add_argument_group("required arguments")
    ra.add_argument("-i", "--input",
                    type=str,
                    required=True,
                    help="Path to a YAML file with AUCMEDI AutoML " + \
                         "configurations")

    # Add optional configuration arguments
    oa = parser_yaml.add_argument_group("optional arguments")
    oa.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")

    # Help page hook for passing no parameters
    if len(sys.argv)==2 and sys.argv[1] == "yaml":
        parser_yaml.print_help(sys.stderr)
        sys.exit(1)

#-----------------------------------------------------#
#                    CLI - Training                   #
#-----------------------------------------------------#
def cli_training(subparsers):
    # Set description for cli training
    desc = """ Pipeline hub for Training via AUCMEDI AutoML """
    # Setup SubParser
    parser_train = subparsers.add_parser("training", help=desc, add_help=False)

    # Add required configuration arguments
    ra = parser_train.add_argument_group("required arguments")
    ra.add_argument("--interface",
                    type=str,
                    required=True,
                    choices=["csv", "directory"],
                    help="String defining format interface for loading/storing"\
                         + " data",
                    )
    ra.add_argument("--path_imagedir",
                    type=str,
                    required=True,
                    help="Path to the directory containing the images",
                    )

    # Add optional configuration arguments
    oa = parser_train.add_argument_group("optional arguments")
    oa.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")
    oa.add_argument("--analysis",
                    type=str,
                    required=False,
                    default="standard",
                    choices=["minimal", "standard", "advanced"],
                    help="Analysis mode for the AutoML training " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--path_data",
                    type=str,
                    required=False,
                    help="Path to the index/class annotation CSV file " + \
                         "(only required for interface 'csv')",
                    )
    oa.add_argument("--path_output",
                    type=str,
                    required=False,
                    default="model",
                    help="Path to the output directory in which fitted " + \
                         "models and metadata are stored " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--ohe",
                    action="store_true",
                    required=False,
                    default=False,
                    help="Boolean option whether annotation data is sparse " + \
                         "categorical or one-hot encoded " + \
                         "(only required for interface 'csv', " + \
                         "default: '%(default)s')",
                    )
    oa.add_argument("--three_dim",
                    action="store_true",
                    required=False,
                    default=False,
                    help="Boolean, whether imaging data is 2D or 3D " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--shape_3D",
                    type=str,
                    required=False,
                    default="128x128x128",
                    help="Desired input shape of 3D volume for architecture "+ \
                         "(will be cropped into, " + \
                         "format: '1x2x3', " + \
                         "default: '%(default)s')",
                    )
    oa.add_argument("--epochs",
                    type=int,
                    required=False,
                    default=500,
                    help="Number of epochs. A single epoch is defined as " + \
                         "one iteration through the complete data set " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--batch_size",
                    type=int,
                    required=False,
                    default=24,
                    help="Number of samples inside a single batch " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--workers",
                    type=int,
                    required=False,
                    default=1,
                    help="Number of workers/threads which preprocess " + \
                         "batches during runtime " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--metalearner",
                    type=str,
                    required=False,
                    default="mean",
                    help="Key for Metalearner or Aggregate function "+ \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--architecture",
                    type=str,
                    required=False,
                    default="DenseNet121",
                    help="Key of single or multiple Architectures " + \
                         "(multiple Architectures are only supported for " + \
                         "'analysis=advanced', " + \
                         "format: 'KEY' or 'KEY,KEY,KEY', " + \
                         "default: '%(default)s')",
                    )

    # Help page hook for passing no parameters
    if len(sys.argv)==2 and sys.argv[1] == "training":
        parser_train.print_help(sys.stderr)
        sys.exit(1)

#-----------------------------------------------------#
#                   CLI - Prediction                  #
#-----------------------------------------------------#
def cli_prediction(subparsers):
    # Set description for cli training
    desc = """ Pipeline hub for Inference via AUCMEDI AutoML """
    # Setup SubParser
    parser_predict = subparsers.add_parser("prediction",
                                           help=desc,
                                           add_help=False)

    # Add required configuration arguments
    ra = parser_predict.add_argument_group("required arguments")
    ra.add_argument("--path_imagedir",
                    type=str,
                    required=True,
                    help="Path to the directory containing the images",
                    )

    # Add optional configuration arguments
    oa = parser_predict.add_argument_group("optional arguments")
    oa.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")
    oa.add_argument("--path_input",
                    type=str,
                    required=False,
                    default="model",
                    help="Path to the input directory in which fitted " + \
                         "models and metadata are stored " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--path_output",
                    type=str,
                    required=False,
                    default="predictions.csv",
                    help="Path to the output file in which predicted csv " + \
                         "file should be stored " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--xai_method",
                    type=str,
                    required=False,
                    help="Key for XAI method " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--xai_directory",
                    type=str,
                    required=False,
                    default="xai",
                    help="Path to the output directory in which predicted " + \
                         "image xai heatmaps should be stored " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--batch_size",
                    type=int,
                    required=False,
                    default=12,
                    help="Number of samples inside a single batch " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--workers",
                    type=int,
                    required=False,
                    default=1,
                    help="Number of workers/threads which preprocess " + \
                         "batches during runtime " + \
                         "(default: '%(default)s')",
                    )

    # Help page hook for passing no parameters
    if len(sys.argv)==2 and sys.argv[1] == "prediction":
        parser_predict.print_help(sys.stderr)
        sys.exit(1)

#-----------------------------------------------------#
#                   CLI - Evaluation                  #
#-----------------------------------------------------#
def cli_evaluation(subparsers):
    # Set description for cli training
    desc = """ Pipeline hub for Evaluation via AUCMEDI AutoML """
    # Setup SubParser
    parser_evaluate = subparsers.add_parser("evaluation",
                                            help=desc,
                                            add_help=False)

    # Add required configuration arguments
    ra = parser_evaluate.add_argument_group("required arguments")
    ra.add_argument("--interface",
                    type=str,
                    required=True,
                    choices=["csv", "directory"],
                    help="String defining format interface for loading/storing"\
                         + " data",
                    )
    ra.add_argument("--path_imagedir",
                    type=str,
                    required=True,
                    help="Path to the directory containing the ground truth" + \
                         " images",
                    )

    # Add optional configuration arguments
    oa = parser_evaluate.add_argument_group("optional arguments")
    oa.add_argument("-h",
                    "--help",
                    action="help",
                    help="show this help message and exit")
    oa.add_argument("--path_data",
                    type=str,
                    required=False,
                    help="Path to the index/class annotation CSV file " + \
                         "(only required for interface 'csv')",
                    )
    oa.add_argument("--ohe",
                    action="store_true",
                    required=False,
                    default=False,
                    help="Boolean option whether annotation data is sparse " + \
                         "categorical or one-hot encoded " + \
                         "(only required for interface 'csv', " + \
                         "default: '%(default)s')",
                    )
    oa.add_argument("--path_input",
                    type=str,
                    required=False,
                    default="predictions.csv",
                    help="Path to the output file in which predicted csv " + \
                         "file are stored " + \
                         "(default: '%(default)s')",
                    )
    oa.add_argument("--path_output",
                    type=str,
                    required=False,
                    default="evaluation",
                    help="Path to the directory in which evaluation " + \
                         "figures and tables should be stored " + \
                         "(default: '%(default)s')",
                    )

    # Help page hook for passing no parameters
    if len(sys.argv)==2 and sys.argv[1] == "evaluation":
        parser_evaluate.print_help(sys.stderr)
        sys.exit(1)
