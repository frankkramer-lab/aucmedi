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
""" State-of-the-art and high-performance medical image classification pipelines
    are heavily utilizing Ensemble Learning strategies.

The idea of Ensemble Learning is to assemble diverse models or multiple predictions and thus
boost prediction performance.

AUCMEDI currently supports the following Ensemble Learning techniques:

| Technique                                  | Description                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| [Augmenting][aucmedi.ensemble.augmenting]  | Inference Augmenting (test-time augmentation) function for augmenting unknown images for prediction.     |
| [Bagging][aucmedi.ensemble.bagging]        | Cross-Validation based Bagging for equal models trained with different sampling.                         |
| [Stacking][aucmedi.ensemble.stacking]      | Ensemble of unequal models with a fitted Metalearner stacked on top of it.                               |
| [Composite][aucmedi.ensemble.composite]    | Combination of Stacking and Bagging via cross-validation with a fitted Metalearner stacked on top of it. |

!!! info
    ![EnsembleLearning_overview](../../images/ensemble.theory.png)

    More information on performance impact of Ensemble Learning in medical image classification can be found here: <br>

    Dominik Müller, Iñaki Soto-Rey, Frank Kramer. (2022)
    An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks.
    arXiv e-print: https://arxiv.org/abs/2201.11440

"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.ensemble.augmenting import predict_augmenting
from aucmedi.ensemble.bagging import Bagging
from aucmedi.ensemble.stacking import Stacking
from aucmedi.ensemble.composite import Composite
