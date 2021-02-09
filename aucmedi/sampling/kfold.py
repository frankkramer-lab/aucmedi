#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
# Internal libraries
from aucmedi.sampling.iterative import MultilabelStratifiedKFold

#-----------------------------------------------------#
#    Function: Sampling via k-fold cross-validation   #
#-----------------------------------------------------#
""" Simple wrapper function for calling k-fold cross-validation sampling functions.
    Allow usage of stratified and iterative sampling algorithm.

    Be aware that multi-label data does not support random stratified sampling.

    Returns:
        List with length n_splits containing tuples with sampled data:
        For example: (train_x, train_y, test_x, test_y)

    Arguments:
        samples (List of Strings):      List of sample/index encoded as Strings.
        labels (NumPy matrix):          NumPy matrix containing the ohe encoded classification.
        n_splits (Integer):             Number of folds (k). Must be at least 2.
        stratified (Boolean):           Option whether to use stratified sampling based on provided labels.
        iterative (Boolean):            Option whether to use iterative sampling algorithm.
        seed (Integer):                 Seed to ensure reproducibility for random functions.
"""
def sampling_kfold(samples, labels, n_splits=3, stratified=True,
                   iterative=False, seed=None):
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

    # Apply sampling and generate folds
    x = np.asarray(samples)
    y = np.asarray(labels)
    for train, test in sampler.split(X=samples, y=wk_labels):
        fold = (x[train], y[train], x[test], y[test])
        results.append(fold)

    # Return result sampling
    return results
