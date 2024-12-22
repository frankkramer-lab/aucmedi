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
import os

# Third Party Libraries
import numpy as np

# Internal Libraries
from aucmedi.data_processing.io_loader import image_loader
from aucmedi.data_processing.subfunctions import Resize
from aucmedi.utils.visualizer import visualize_heatmap
from aucmedi.xai.methods import xai_dict


#-----------------------------------------------------#
#                    XAI - Decoder                    #
#-----------------------------------------------------#
def xai_decoder(data_gen, model, preds=None, method="gradcam", layerName=None,
                alpha=0.4, out_path=None):
    """ XAI Decoder function for automatic computation of Explainable AI heatmaps.

    This module allows to visualize which regions were crucial for the neural network model
    to compute a classification on the provided unknown images.

    - If `out_path` parameter is None, heatmaps are returned as NumPy array.
    - If a path is provided as `out_path`, then heatmaps are stored to disk as PNG files.

    ???+ info "XAI Methods"
        The XAI Decoder can be run with different XAI methods as backbone.

        A list of all implemented methods and their keys can be found here: <br>
        [aucmedi.xai.methods][]

    ???+ example "Example"
        ```python
        # Create a DataGenerator for data I/O
        datagen = DataGenerator(samples[:3], "images_xray/", labels=None, resize=(299, 299))

        # Get a model
        model = NeuralNetwork(n_labels=3, channels=3, architecture="Xception",
                               input_shape=(299,299))
        model.load("model.xray.keras")

        # Make some predictions
        preds = model.predict(datagen)

        # Compute XAI heatmaps via Grad-CAM (resulting heatmaps are stored in out_path)
        xai_decoder(datagen, model, preds, method="gradcam", out_path="xai.xray_gradcam")
        ```

    Args:
        data_gen (DataGenerator):           A data generator which will be used for inference.
        model (NeuralNetwork):              Instance of a AUCMEDI neural network class.
        preds (numpy.ndarray):              NumPy Array of classification prediction encoded as OHE (output of a AUCMEDI
                                            prediction).
        method (str):                       XAI method class instance or index. By default, GradCAM is used as XAI
                                            method.
        layerName (str):                    Layer name of the convolutional layer for heatmap computation. If `None`,
                                            the last conv layer is used.
        alpha (float):                      Transparency value for heatmap overlap plotting on input image
                                            (range: [0-1]).
        out_path (str):                     Output path in which heatmaps are saved to disk as PNG files.

    Returns:
        images (numpy.ndarray):             Combined array of images. Will be only returned if `out_path` parameter is
                                            `None`.
        heatmaps (numpy.ndarray):           Combined array of XAI heatmaps. Will be only returned if `out_path`
                                            parameter is `None`.
    """
    # Initialize & access some variables
    n_classes = model.n_labels
    sample_list = data_gen.samples
    # Prepare XAI output methods
    res_img = []
    res_xai = []
    if out_path is not None and not os.path.exists(out_path):
        os.mkdir(out_path)
    # Initialize xai method
    if isinstance(method, str) and method in xai_dict:
        xai_method = xai_dict[method](model.model, layerName=layerName)
    else:
        xai_method = method

    # Iterate over all samples
    for i in range(0, len(sample_list)):
        # Load original image
        img_org = image_loader(sample_list[i], data_gen.path_imagedir,
                               image_format=data_gen.image_format,
                               grayscale=data_gen.grayscale)
        shape_org = img_org.shape[0:2]
        # Load processed image
        img_prc = data_gen.preprocess_image(i)
        img_batch = np.expand_dims(img_prc, axis=0)
        # If preds given, compute heatmap only for argmax class
        if preds is not None:
            ci = np.argmax(preds[i])
            xai_map = xai_method.compute_heatmap(img_batch, class_index=ci)
            xai_map = Resize(shape=shape_org).transform(xai_map)
            postprocess_output(sample_list[i], img_org, xai_map, n_classes,
                               data_gen, res_img, res_xai, out_path, alpha)
        # If no preds given, compute heatmap for all classes
        else:
            sample_maps = []
            for ci in range(0, n_classes):
                xai_map = xai_method.compute_heatmap(img_batch, class_index=ci)
                xai_map = Resize(shape=shape_org).transform(xai_map)
                sample_maps.append(xai_map)
            sample_maps = np.array(sample_maps)
            postprocess_output(sample_list[i], img_org, sample_maps, n_classes,
                               data_gen, res_img, res_xai, out_path, alpha)
    # Return output directly if no output path is defined (and convert to NumPy)
    if out_path is None: return np.array(res_img), np.array(res_xai)


#-----------------------------------------------------#
#          Subroutine: Output Postprocessing          #
#-----------------------------------------------------#
def postprocess_output(sample, image, xai_map, n_classes, data_gen,
                       res_img, res_xai, out_path, alpha):
    """ Helper/Subroutine function for XAI Decoder.

    Caches heatmap for direct output or generates a visualization as PNG.
    """
    # Update result lists for direct output
    if out_path is None:
        res_img.append(image)
        res_xai.append(xai_map)
    # Generate XAI heatmap visualization
    else:
        # Create XAI path
        if data_gen.image_format:
            xai_file = sample + "." + data_gen.image_format
        else:
            xai_file = sample
        if os.sep in xai_file: xai_file = xai_file.replace(os.sep, ".")
        path_xai = os.path.join(out_path, xai_file)
        # If preds given, output only argmax class heatmap
        if len(xai_map.shape) == 2:
            visualize_heatmap(image, xai_map, out_path=path_xai, alpha=alpha)
        # If no preds given, output heatmaps for all classes
        else:
            for c in range(0, n_classes):
                path_xai_c = path_xai[:-4] + ".class_" + str(c) + \
                             path_xai[-4:]
                visualize_heatmap(image, xai_map[c], out_path=path_xai_c,
                                  alpha=alpha)
