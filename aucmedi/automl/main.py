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
import pathlib
# Internal libraries


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
    subparsers = parser.add_subparsers(title="Application Modes")
    # Return parsers
    return parser, subparsers

#-----------------------------------------------------#
#                     CLI - YAML                      #
#-----------------------------------------------------#
def cli_yaml(subparsers):
    # Set description for cli training
    desc = """ YAML interface for reading configurations from a file """
    # Setup SubParser
    parser_yaml = subparsers.add_parser("yaml", help=desc)

    # Add configuration arguments
    parser_yaml.add_argument("-i", "--input", type=str,
                             help="Path to a YAML file with AUCMEDI AutoML " + \
                                  "configurations")

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
                    required=True,
                    default="standard",
                    choices=["minimal", "standard", "composite"],
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

    # shape_3D (tuple of int):            Desired input shape of 3D volume for architecture (will be cropped).
    # epochs (int):                       Number of epochs. A single epoch is defined as one iteration through
    #                                     the complete data set.
    # batch_size (int):                   Number of samples inside a single batch.
    # workers (int):                      Number of workers/threads which preprocess batches during runtime.
    # metalearner (str):                  Key for Metalearner or Aggregate function.
    # architecture (str or list of str):  Key (str) of a neural network model Architecture class instance.

#-----------------------------------------------------#
#                   CLI - Prediction                  #
#-----------------------------------------------------#
def cli_prediction(subparsers):
    # Set description for cli training
    desc = """ Pipeline hub for Inference via AUCMEDI AutoML """
    # Setup SubParser
    pp = subparsers.add_parser("prediction", help=desc)

    pp.add_argument('--path_pred', choices='XYZ', help='baz help')

    # # Define SubParser Evaluation
    # parser_pred = subparsers.add_parser('evaluation', help='asd')
    # parser_pred.add_argument('--path_pred', choices='XYZ', help='baz help')

#-----------------------------------------------------#
#                Main Method - Runner                 #
#-----------------------------------------------------#
if __name__ == "__main__":
    # Initialize argparser core
    parser, subparsers = cli_core()
    # Define Subparser YAML
    cli_yaml(subparsers)
    # Define Subparser Training
    cli_training(subparsers)
    # Define Subparser Prediction
    cli_prediction(subparsers)

    # Help page hook for passing no parameters
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # Parse arguments
    else : parser.parse_args()


# read any single paramter via args -> 3x groups (analysis) with various parameters each

# read a single yaml file -> one path to yaml file

# run associated modi:
# simple
# standard
# advanced (composite)
