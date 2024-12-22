# ==============================================================================#
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
# ==============================================================================#
# -----------------------------------------------------#
#                    Documentation                    #
# -----------------------------------------------------#
""" AUCMEDI AutoML configuration parsers. Allows the usage of AUCMEDI's AutoML 
pipeline with configuration files instead of the commandline arguments."""


# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
from aucmedi.automl.config_parsers.parser_cli import parse_cli
from aucmedi.automl.config_parsers.config_file_parser import parse_config_file
from aucmedi.automl.config_parsers.validation_classes import (
    BaseConfig,
    TrainingConfig,
    PredictionConfig,
    EvaluationConfig
)


__all__ = [
    "parse_config_file",
    "parse_cli",
    "BaseConfig",
    "TrainingConfig",
    "PredictionConfig",
    "EvaluationConfig"
]