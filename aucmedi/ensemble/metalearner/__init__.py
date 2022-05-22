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
""" Library of implemented Metalearners in AUCMEDI.

A Metalearner can be passed to an ensemble like Stacking and merges multiple class
predictions into a single prediction.

Metalearner are similar to [aggregate()][aucmedi.ensemble.aggregate] functions,
however Metalearners are models which require fitting before usage.

```
Ensembled predictions encoded in a NumPy Matrix with shape (N_models, N_classes).
Example: [[0.5, 0.4, 0.1],
          [0.4, 0.3, 0.3],
          [0.5, 0.2, 0.3]]
-> shape (3, 3)

Merged prediction encoded in a NumPy Matrix with shape (1, N_classes).
Example: [[0.4, 0.3, 0.3]]
-> shape (1, 3)
```

Metalearners are based on the abstract base class [Metalearner_Base][aucmedi.ensemble.metalearner.ml_base],
which allow simple integration of custom Metalearners for Ensemble.
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# Import metalearners
from aucmedi.ensemble.metalearner.logistic_regression import Logistic_Regression
from aucmedi.ensemble.metalearner.naive_bayes import Naive_Bayes

#-----------------------------------------------------#
#           Access Functions to Metalearners          #
#-----------------------------------------------------#
metalearner_dict = {"logistic_regression": Logistic_Regression,
                    "naive_bayes": Naive_Bayes,
}
""" Dictionary of implemented Metalearners. """
