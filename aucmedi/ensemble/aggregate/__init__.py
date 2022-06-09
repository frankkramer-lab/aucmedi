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
""" Library of implemented Aggregate functions in AUCMEDI.

An Aggregate function can be passed to an ensemble and merges multiple class predictions
into a single prediction.

```
Assembled predictions encoded in a NumPy matrix with shape (N_models, N_classes).
Example: [[0.5, 0.4, 0.1],
          [0.4, 0.3, 0.3],
          [0.5, 0.2, 0.3]]
-> shape (3, 3)

Merged prediction encoded in a NumPy matrix with shape (1, N_classes).
Example: [[0.4, 0.3, 0.3]]
-> shape (1, 3)
```

???+ example "Example"
    ```python
    # Recommended: Apply an Ensemble like Augmenting (test-time augmentation) with Majority Vote
    preds = predict_augmenting(model, test_datagen, n_cycles=5, aggregate="majority_vote")

    # Manual: Apply an Ensemble like Augmenting (test-time augmentation) with Majority Vote
    from aucmedi.ensemble.aggregate import MajorityVote
    my_agg = MajorityVote()
    preds = predict_augmenting(model, test_datagen, n_cycles=5, aggregate=my_agg)
    ```

Aggregate functions are based on the abstract base class [Aggregate_Base][aucmedi.ensemble.aggregate.agg_base.Aggregate_Base],
which allows simple integration of custom aggregate methods for Ensemble.
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# Import aggregate functions
from aucmedi.ensemble.aggregate.averaging_mean import AveragingMean
from aucmedi.ensemble.aggregate.averaging_median import AveragingMedian
from aucmedi.ensemble.aggregate.majority_vote import MajorityVote
from aucmedi.ensemble.aggregate.softmax import Softmax
from aucmedi.ensemble.aggregate.global_argmax import GlobalArgmax

#-----------------------------------------------------#
#       Access Functions to Aggregate Functions       #
#-----------------------------------------------------#
aggregate_dict = {"mean": AveragingMean,
                  "median": AveragingMedian,
                  "majority_vote": MajorityVote,
                  "softmax": Softmax,
                  "global_argmax": GlobalArgmax,
}
""" Dictionary of implemented Aggregate functions. """
