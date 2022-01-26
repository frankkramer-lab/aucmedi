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
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import numpy as np
import json
import pandas as pd

#-----------------------------------------------------#
#         Data Loader Interface based on JSON         #
#-----------------------------------------------------#
""" Data Input Interface for loading a dataset via a JSON and an image directory.
    This function allow simple parsing of class annotations encoded in a JSON.

    Format Sparse:
        - Name Index (key) : Class (value)

    Format One-Hot Encoded:
        - Name Index (key) : List consisting of binary integers.

Arguments:
    path_data (String):                     Path to the json file.
    path_imagedir (String):                 Path to the directory containing the images.
    allowed_image_formats (String list):    List of allowed imaging formats. (provided by IO_Interface)
    training (Boolean):                     Boolean option whether annotation data is available.
    ohe (Boolean):                          Boolean option whether annotation data is sparse categorical or one-hot encoded.
"""
def json_loader(path_data, path_imagedir, allowed_image_formats, training=True,
                ohe=True):
    # Load JSON file
    with open(path_data, "r") as json_reader:
        dt_json = json.load(json_reader)
    # Identify image format by peaking first image
    image_format = None
    for file in os.listdir(path_imagedir):
        format = file.split(".")[-1]
        if format.lower() in allowed_image_formats or \
           format.upper() in allowed_image_formats:
           image_format = format
           break
    # Raise Exception if image format is unknown
    if image_format is None:
        raise Exception("Unknown image format.", path_imagedir)

    # Verify if all images are existing
    lever = True
    for sample in dt_json:
        if sample == "legend" : continue
        # Check if image ending is already in sample name by peaking first one
        if lever:
            lever = False
            if sample.endswith("." + image_format) : image_format = None
        # Obtain image file path
        if image_format : img_file = sample + "." + image_format
        else : img_file = sample
        path_img = os.path.join(path_imagedir, img_file)
        # Check existance
        if not os.path.exists(path_img):
            raise Exception("Image does not exist / not accessible!",
                            'Sample: "' + sample + '"', path_img)

    # If JSON is for inference (no annotation data)
    if not training:
        # Ensure index list to contain strings
        if "legend" in dt_json : del dt_json["legend"]
        index_list = [str(x) for x in dt_json]
        # -> return parsing
        return index_list, None, None, None, image_format

    # Try parsing with a sparse categorical class format
    if not ohe:
        # Parse class name information
        if "legend" in dt_json:
            class_names = dt_json["legend"]
            del dt_json["legend"]
        else : class_names = None
        # Obtain class information and index list
        index_list = []
        classes_sparse = []
        for sample in dt_json:
            index_list.append(str(sample))
            classes_sparse.append(dt_json[sample])
        if class_names is None : class_names = np.unique(classes_sparse).tolist()
        class_n = len(class_names)
        # Parse sparse categorical annotations to One-Hot Encoding
        class_ohe = pd.get_dummies(classes_sparse).to_numpy()
    # Try parsing one-hot encoded format
    else:
        # Parse information
        if "legend" in dt_json:
            class_names = dt_json["legend"]
            del dt_json["legend"]
            class_n = len(class_names)
        else:
            class_names = None
            class_n = None
        # Obtain class information and index list
        index_list = []
        class_data = []
        for sample in dt_json:
            index_list.append(str(sample))
            class_data.append(dt_json[sample])
        class_ohe = np.array(class_data)
        # Verify number of class annotation
        if class_n is None : class_ohe.shape[1]

    # Validate if number of samples and number of annotations match
    if len(index_list) != len(class_ohe):
        raise Exception("Number of samples and annotation does not match!",
                        len(index_list), len(class_ohe))

    # Return parsed JSON data
    return index_list, class_ohe, class_n, class_names, image_format
