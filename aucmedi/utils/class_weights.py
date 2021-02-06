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
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np

#-----------------------------------------------------#
#               Class Weight Computation              #
#-----------------------------------------------------#
""" Simple wrapper function for scikit learn class_weight function.
    The class weights can be used for weighting the loss function on imbalanced data.
    Returned is a class weight dictionary which can be directly feeded in Keras fit().

    NumPy array shape has to be (n_samples, n_classes) like this: (500, 4).

    Scikit learn class_weight function:
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    Arguments:
        ohe_array (NumPy matrix):       NumPy matrix containing the ohe encoded classification.
        method (String):                Dictionary or modus, how class weights should be computed.
"""
def compute_class_weights(ohe_array, method="balanced"):
    # Obtain sparse categorical array and number of classes
    class_array = np.argmax(ohe_array, axis=-1)
    n_classes = np.unique(class_array)
    # Compute class weights with scikit learn
    class_weights = compute_class_weight(class_weight=method, classes=n_classes,
                                         y=class_array)
    # Convert class weight array to dictionary
    class_weights_dict = dict(enumerate(class_weights))
    # Return resulting class weight dictionary
    return class_weights_dict

#-----------------------------------------------------#
#              Sample Weight Computation              #
#-----------------------------------------------------#
""" Simple wrapper function for scikit learn sample_weight function.
    The sample weights can be used for weighting the loss function on imbalanced data.
    Returned sample weight array which can be directly feeded to a AUCMEDi DataGenerator.

    NumPy array shape has to be (n_samples, n_classes) like this: (500, 4).

    Scikit learn sample_weight function:
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html

    Arguments:
        ohe_array (NumPy matrix):       NumPy matrix containing the ohe encoded classification.
        method (String):                Dictionary or modus, how class weights should be computed.
"""
def compute_sample_weights(ohe_array, method="balanced"):
    # Compute sample weights with scikit learn
    sample_weights = compute_sample_weight(class_weight=method, y=ohe_array)
    # Return resulting sample weights
    return sample_weights
