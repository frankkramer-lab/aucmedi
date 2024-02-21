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
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np

#-----------------------------------------------------#
#               Class Weight Computation              #
#-----------------------------------------------------#
def compute_class_weights(ohe_array, method="balanced"):
    """ Simple wrapper function for scikit learn class_weight function.

    The class weights can be used for weighting the loss function on imbalanced data.

    ???+ info
        NumPy array shape has to be (n_samples, n_classes) like this: (500, 4).

    ???+ example "Example"
        ```python
        # Compute class weights
        cw_loss, cw_fit = compute_class_weights(class_ohe)

        # Provide class weights to loss function
        model = NeuralNetwork(nclasses, channels=3, loss=categorical_focal_loss(cw_loss))

        # Provide class weights to keras fit()
        model.train(index_list, epochs=50, class_weights=cw_fit)
        ```

    ??? abstract "Based on Reference"
        Scikit-learn class_weight function: <br>
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html  <br>

    Args:
        ohe_array (numpy.ndarray):  NumPy matrix containing the ohe encoded classification.
        method (str):               Dictionary or modus, how class weights should be computed.

    Returns:
        class_weights_list (numpy.ndarray):     Class weight list which can be feeded to a loss function.
        class_weights_dict (dict):              Class weight dictionary which can be feeded to
                                                [train()][aucmedi.neural_network.model.NeuralNetwork.train]
                                                or keras.model.fit().
    """
    # Obtain sparse categorical array and number of classes
    class_array = np.argmax(ohe_array, axis=-1)
    n_classes = np.unique(class_array)
    # Compute class weights with scikit learn
    class_weights_list = compute_class_weight(class_weight=method,
                                              classes=n_classes, y=class_array)
    # Convert class weight array to dictionary
    class_weights_dict = dict(enumerate(class_weights_list))
    # Return resulting class weights as list and dictionary
    return class_weights_list, class_weights_dict

#-----------------------------------------------------#
#           Multi-Label Weight Computation            #
#-----------------------------------------------------#
def compute_multilabel_weights(ohe_array, method="balanced"):
    """ Function for computing class weights individually for multi-label data.

    Class weights can be used for weighting the loss function on imbalanced data.
    Returned is a class weight list which can be passed to loss functions.

    ???+ info
        NumPy array shape has to be (n_samples, n_classes) like this: (500, 4).

    ??? abstract "Based on Reference"
        Class weight computation is based on Scikit learn class_weight function: <br>
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html  <br>

    Args:
        ohe_array (numpy.ndarray):      NumPy matrix containing the ohe encoded classification.
        method (str):                   Dictionary or modus, how class weights should be computed.

    Returns:
        class_weights (numpy.ndarray):      Class weight list which can be fed to a loss function.
    """
    # Identify number of classes
    n_classes = np.shape(ohe_array)[1]
    # Initialize class weight list
    class_weights = np.empty([n_classes])
    # Compute weight for each class individually
    for i in range(0, n_classes):
        weight = compute_class_weight(class_weight=method, classes=np.array([0,1]),
                                      y=ohe_array[:, i])
        class_weights[i] = weight[1]
    # Return resulting class weight list
    return class_weights

#-----------------------------------------------------#
#              Sample Weight Computation              #
#-----------------------------------------------------#
def compute_sample_weights(ohe_array, method="balanced"):
    """ Simple wrapper function for scikit learn sample_weight function.

    The sample weights can be used for weighting the loss function on imbalanced data.
    Returned sample weight array which can be directly fed to an AUCMEDI [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    ???+ info
        NumPy array shape has to be (n_samples, n_classes) like this: (500, 4).

    ??? abstract "Based on Reference"
        Scikit learn sample_weight function: <br>
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html <br>

    Args:
        ohe_array (numpy.ndarray):      NumPy matrix containing the ohe encoded classification.
        method (str):                   Dictionary or modus, how class weights should be computed.

    Returns:
        sample_weights (numpy.ndarray):     Sample weight list which can be fed to an AUCMEDI
                                            [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].
    """
    # Compute sample weights with scikit learn
    sample_weights = compute_sample_weight(class_weight=method, y=ohe_array)
    # Return resulting sample weights
    return sample_weights
