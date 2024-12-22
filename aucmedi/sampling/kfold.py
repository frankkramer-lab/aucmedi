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
# Python Standard Library

# Third Party Libraries
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

# Internal Libraries
from aucmedi.sampling.iterative import MultilabelStratifiedKFold


#-----------------------------------------------------#
#    Function: Sampling via k-fold cross-validation   #
#-----------------------------------------------------#
def sampling_kfold(samples, labels, metadata=None, n_splits=3,
                   stratified=True, iterative=False, seed=None):
    """ Simple wrapper function for calling k-fold cross-validation sampling functions.

    Allow usage of stratified and iterative sampling algorithm.

    ???+ warning
        Be aware that multi-label data does not support random stratified sampling.

    ???+ example
        The sampling is returned as list with length n_splits containing tuples with sampled data.

        ```python title="Example for n_splits=3"
        cv = sampling_kfold(samples, labels, n_splits=3)

        # sampling in which x = samples and y = labels
        # cv <-> [(train_x, train_y, test_x, test_y),   # fold 1
        #         (train_x, train_y, test_x, test_y),   # fold 2
        #         (train_x, train_y, test_x, test_y)]   # fold 3

        # Recommended access on the folds
        for fold in cv:
            (train_x, train_y, test_x, test_y) = fold
        ```

        ```python title="Example with metadata"
        cv = sampling_kfold(samples, labels, metadata, n_splits=2)

        # sampling in which x = samples, y = labels and m = metadata
        # cv <-> [(train_x, train_y, train_m, test_x, test_y, test_m),      # fold 1
        #         (train_x, train_y, train_m, test_x, test_y, test_m)]      # fold 2
        ```

    Args:
        samples (list of str):      List of sample/index encoded as Strings.
        labels (numpy.ndarray):     NumPy matrix containing the ohe encoded classification.
        metadata (numpy.ndarray):   NumPy matrix with additional metadata. Have to be shape (n_samples, meta_variables).
        n_splits (int):             Number of folds (k). Must be at least 2.
        stratified (bool):          Option whether to use stratified sampling based on provided labels.
        iterative (bool):           Option whether to use iterative sampling algorithm.
        seed (int):                 Seed to ensure reproducibility for random functions.

    Returns:
        sampling (list of tuple):   List with length `n_splits` containing tuples with sampled data.
    """
    # Initialize variables
    results = []
    wk_labels = labels

    # Initialize random sampler
    if not stratified and not iterative:
        sampler = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Initialize random stratified sampler
    elif stratified and not iterative:
        sampler = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed)
        wk_labels = np.argmax(wk_labels, axis=-1)
    # Initialize iterative stratified sampler
    else:
        sampler = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True,
                                            random_state=seed)

    # Preprocess data
    x = np.asarray(samples)
    y = np.asarray(labels)
    if metadata is not None: m = np.asarray(metadata)

    # Apply sampling and generate folds
    for train, test in sampler.split(X=samples, y=wk_labels):
        # Simple sampling
        if metadata is None:
            fold = (x[train], y[train], x[test], y[test])
        # Sampling with metadata
        else:
            fold = (x[train], y[train], m[train], x[test], y[test], m[test])
        results.append(fold)

    # Return result sampling
    return results
