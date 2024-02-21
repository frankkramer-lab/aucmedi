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
""" API reference for the AUCMEDI AutoML pipeline.

The mentality behind AutoML is to ensure easy application, integration and
maintenance of complex medical image classification pipelines.
AUCMEDI provides a fast and intuitive interface through AUCMEDI AutoML for building,
application and sharing of state-of-the-art medical image classification models.

The AutoML pipelines are categorized into the following modes:
`training`, `prediction` and `evaluation`.

- The console entry `aucmedi` refers to [aucmedi.automl.main:main][aucmedi.automl.main].
- The Argparse interface for CLI is defined in [aucmedi.automl.cli][aucmedi.automl.cli]
- Each AutoML mode is implemented as a code block defining the AUCMEDI pipeline.

!!! info
    | AutoML Mode   | Argparse                                              | Code Block (Pipeline)                         |
    | ------------- | ----------------------------------------------------- | --------------------------------------------- |
    | `training`    | [CLI - Training][aucmedi.automl.cli.cli_training]     | [Block - Train][aucmedi.automl.block_train]   |
    | `prediction`  | [CLI - Prediction][aucmedi.automl.cli.cli_prediction] | [Block - Predict][aucmedi.automl.block_pred]  |
    | `evaluation`  | [CLI - Evaluation][aucmedi.automl.cli.cli_evaluation] | [Block - Evaluate][aucmedi.automl.block_eval] |

More information can be found in the docs: [Documentation - AutoML](../../automl/overview/)
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# Block functions
from aucmedi.automl.block_train import block_train
from aucmedi.automl.block_pred import block_predict
from aucmedi.automl.block_eval import block_evaluate
# Parser
from aucmedi.automl.parser_yaml import parse_yaml
from aucmedi.automl.parser_cli import parse_cli
