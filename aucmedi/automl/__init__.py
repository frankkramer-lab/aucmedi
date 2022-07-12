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
#                    Documentation                    #
#-----------------------------------------------------#
""" Work in Progress.

structure explaination

ref to automl docs
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
