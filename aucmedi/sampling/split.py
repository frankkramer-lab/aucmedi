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
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
# Internal libraries
from aucmedi.sampling.iterative import MultilabelStratifiedShuffleSplit

#-----------------------------------------------------#
#       Function: Sampling via Percentage Split       #
#-----------------------------------------------------#
def sampling_split(samples, labels, metadata=None, sampling=[0.8, 0.2],
                   stratified=True, iterative=False, seed=None):
    """ Simple wrapper function for calling percentage split sampling functions.

    Allow usage of stratified and iterative sampling algorithm.

    ???+ warning
        Be aware that multi-label data does not support random stratified sampling.

    Percentage split ratios have to be provided with a sampling list.
    Each percentage value in the list defines the approximate split size.
    Sum of percentage split ratios have to equal 1!

    ???+ example
        ```python title="Example for common train/val/test split"
        split_ratio = [0.7, 0.25, 0.05]
        ds = sampling_split(samples, labels, sampling=split_ratio)

        # Returns a list with the following elements as tuples:
        print(ds[0])  # -> (samples_a, labels_a)                with 70% of complete dataset
        print(ds[1])  # -> (samples_b, labels_b)                with 25% of complete dataset
        print(ds[2])  # -> (samples_c, labels_c)                with  5% of complete dataset
        ```

        ```python title="Example with metadata"
        ds = sampling_split(samples, labels, metadata, sampling=[0.8, 0.2])

        # Returns a list with the following elements as tuples:
        print(ds[0])  # -> (samples_a, labels_a, metadata_a)    with 80% of complete dataset
        print(ds[1])  # -> (samples_b, labels_b, metadata_b)    with 20% of complete dataset
        ```

    Args:
        samples (list of str):          List of sample/index encoded as Strings.
        labels (numpy.ndarray):         NumPy matrix containing the ohe encoded classification.
        metadata (numpy.ndarray):       NumPy matrix with additional metadata. Have to be shape (n_samples, meta_variables).
        sampling (list of float):       List of percentage values with split sizes.
        stratified (bool):              Option whether to use stratified sampling based on provided labels.
        iterative (bool):               Option whether to use iterative sampling algorithm.
        seed (int):                     Seed to ensure reproducibility for random functions.

    Returns:
        results (list of tuple):        List with `len(sampling)` containing tuples with sampled data:
                                        (samples_a, labels_a) and with metadata (samples_a, labels_a, metadata_a).
    """
    # Verify sampling percentages
    if not np.isclose(sum(sampling), 1.0):
        raise ValueError("Sum of Percentage split ratios as sampling do not" + \
                         " equal 1", sampling, np.sum(sampling))
    # Initialize leftover with the complete dataset
    leftover_samples = np.asarray(samples)
    leftover_labels = np.asarray(labels)
    if metadata is not None : leftover_meta = np.asarray(metadata)
    leftover_p = 0.0
    # Initialize result list
    results = []

    # Perform sampling for each percentage split
    for i in range(0, len(sampling)):
        # For last split, just take leftover data as subset
        if i == len(sampling)-1:
            # Generate split
            if metadata is None : split = (leftover_samples, leftover_labels)
            else : split = (leftover_samples, leftover_labels, leftover_meta)
            # Append splitted data and stop
            results.append(split)
            break

        # Identify split percentage for remaining data
        p = sampling[i] / (1.0 - leftover_p)
        # Initialize random sampler
        if not stratified and not iterative:
            sampler = ShuffleSplit(n_splits=1, random_state=seed,
                                   train_size=(1.0-p), test_size=p)
        # Initialize random stratified sampler
        elif stratified and not iterative:
            sampler = StratifiedShuffleSplit(n_splits=1, random_state=seed,
                                             train_size=(1.0-p), test_size=p)
        # Initialize iterative stratified sampler
        else:
            sampler = MultilabelStratifiedShuffleSplit(n_splits=1,
                            random_state=seed, train_size=(1.0-p), test_size=p)

        # Apply sampling
        subset_generator = sampler.split(X=leftover_samples, y=leftover_labels)
        subsets = next(subset_generator)
        # Generate split
        if metadata is None:
            split = (leftover_samples[subsets[1]], leftover_labels[subsets[1]])
        else : split = (leftover_samples[subsets[1]],
                        leftover_labels[subsets[1]],
                        leftover_meta[subsets[1]])
        # Append splitted data
        results.append(split)
        # Update remaining data
        leftover_p += sampling[i]
        leftover_samples = leftover_samples[subsets[0]]
        leftover_labels = leftover_labels[subsets[0]]
        if metadata is not None : leftover_meta = leftover_meta[subsets[0]]

    # Return result sampling
    return results
