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
""" Entry script (runner/main function) which will be called in AUCMEDI AutoML.

The console entry `aucmedi` refers to `aucmedi.automl.main:main`.

Executes AUCMEDI AutoML pipeline for training, prediction and evaluation.

More information can be found in the docs: [Documentation - AutoML](../../../automl/overview/)
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import sys
# Internal libraries
from aucmedi.automl import *
from aucmedi.automl.cli import *

#-----------------------------------------------------#
#                Main Method - Runner                 #
#-----------------------------------------------------#
def main():
    # Initialize argparser core
    parser, subparsers = cli_core()
    # # Define Subparser YAML
    # cli_yaml(subparsers)
    # Define Subparser Training
    cli_training(subparsers)
    # Define Subparser Prediction
    cli_prediction(subparsers)
    # Define Subparser Evaluation
    cli_evaluation(subparsers)

    # Help page hook for passing no parameters
    if len(sys.argv)<=1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # Parse arguments
    else : args = parser.parse_args()

    # Call corresponding cli or yaml parser
    if args.hub == "yaml" : config = parse_yaml(args)
    else : config = parse_cli(args)

    # Run training pipeline
    if config["hub"] == "training" : block_train(config)
    # Run prediction pipeline
    if config["hub"] == "prediction" : block_predict(config)
    # Run evaluation pipeline
    if config["hub"] == "evaluation" : block_evaluate(config)

# Runner for direct script call
if __name__ == "__main__":
    main()
