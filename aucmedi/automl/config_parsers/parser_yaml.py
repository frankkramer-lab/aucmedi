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
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
# Third Party Libraries
from pydantic import BaseModel
# Internal libraries

#-----------------------------------------------------#
#              Base Models for Validation             #
#-----------------------------------------------------#
class BaseConfig(BaseModel):
    hub: str
    path_imagedir: str


class TrainingConfig(BaseModel):
    path_modeldir: str
    path_gt: str | None
    ohe: bool
    analysis: str
    three_dim: bool
    shape_3D: list[int]
    epochs: int
    batch_size: int
    workers: int
    metalearner: str
    architecture: list[str]


class PredictionConfig(BaseModel):
    path_modeldir: str
    path_pred: str
    xai_method: str | None
    xai_directory: str
    batch_size: int
    workers: int


class EvaluationConfig(BaseModel):
    path_gt: str | None
    ohe: bool
    path_pred: str
    path_evaldir: str
