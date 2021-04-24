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
# External Libraries
import numpy as np
# AUCMEDI Libraries
from aucmedi.xai import xai_dict

#-----------------------------------------------------#
#                    XAI - Decoder                    #
#-----------------------------------------------------#
""" XAI Decoder function for automatic computation of Explainable AI heatmaps.
    This module allows to visualize which regions were crucial for the neural network model
    to compute a classification on the provided unknown images.

XAI Methods:
    The XAI Decoder can be run with different XAI methods as backbone.
    List of XAI Methods : ["gradcam"]

Arguments:
    data_gen (DataGenerator):           A data generator which will be used for inference.
    model (Neural_Network):             Instance of a AUCMEDI neural network class.
    preds (NumPy Array):                NumPy Array of classification prediction encoded as OHE (output of a AUCMEDI prediction).
    method (String):                    XAI method class instance or index. By default, GradCAM is used as XAI method.
"""
def xai_decoder(data_gen, model, preds=None, method="gradcam"):
    # Initialize & access some variables
    batch_size = data_gen.batch_size
    n_classes = model.n_labels
    res = []
    pos = 0
    # Initialize xai method
    if isinstance(method, str) and method in xai_dict:
        xai_method = xai_dict[method](model.model)
    else : xai_method = method

    # Iterate over all samples
    for i in range(0, len(data_gen)):
        # Obtain imaging batch from the data generator
        batch = next(data_gen)[0]
        # Process images
        for j in range(0, len(batch)):
            image = batch[[j]]
            # If preds given, compute heatmap only for argmax
            if preds is not None:
                ci = np.argmax(preds[pos])
                xai_map = xai_method.compute_heatmap(image, class_index=ci)
                res.append(xai_map)
                pos += 1
            # If no preds given, compute heatmap for all classes
            else:
                sample_maps = []
                for ci in range(0, n_classes):
                    xai_map = xai_method.compute_heatmap(image, class_index=ci)
                    sample_maps.append(xai_map)
                res.append(sample_maps)
    # Transform to NumPy and return results
    return np.array(res)
