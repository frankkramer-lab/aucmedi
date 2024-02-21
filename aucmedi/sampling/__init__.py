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
""" Interface for k-fold and percentage-split sampling functions in AUCMEDI.

| Function                   | Description                          |
| -------------------------- | ------------------------------------ |
| [aucmedi.sampling.split][] | Simple wrapper function for calling percentage split sampling functions.        |
| [aucmedi.sampling.kfold][] | Simple wrapper function for calling k-fold cross-validation sampling functions. |

???+ example "Recommended Import"
    ```python
    # Import sampling
    from aucmedi.sampling import sampling_split, sampling_kfold

    # Run selected sampling
    ds_ps = sampling_split(samples, labels, sampling=[0.7, 0.25, 0.05])
    ds_ks = sampling_kfold(samples, labels, n_splits=3)
    ```
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.sampling.split import sampling_split
from aucmedi.sampling.kfold import sampling_kfold
