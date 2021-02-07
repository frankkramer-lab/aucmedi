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
import os
import numpy as np
import pandas as pd

#-----------------------------------------------------#
#          Data Loader Interface based on CSV         #
#-----------------------------------------------------#
""" Data Input Interface for loading a dataset via a CSV and an image directory.
    This function allow simple parsing of class annotations encoded in a CSV.

    CSV Format 1:
        - Name Column: "SAMPLE" -> String Value
        - Class Column: "CLASS" -> Sparse Categorical Classes (String/Integers)
        - Optional Meta Columns possible

    CSV Format 2:
        - Name Column: "SAMPLE"
        - One-Hot Encoded Class Columns:
            -> If OHE parameter provides list of column names -> use these
            -> Else try to use all other columns as OHE columns
        - Optional Meta Columns only possible if OHE parameter provided

Arguments:
    path_data (String):                     Path to the csv file.
    path_imagedir (String):                 Path to the directory containing the images.
    allowed_image_formats (String list):    List of allowed imaging formats. (provided by IO_Interface)
    training (Boolean):                     Boolean option whether annotation data is available.
    ohe (Boolean):                          Boolean option whether annotation data is sparse categorical or one-hot encoded.
    ohe_range (String list):                List of column name values if annotation encoded in OHE. Example: ["classA", "classB", "classC"]
    col_sample (String):                    Index column name for the sample name column. Default: 'SAMPLE'
    col_class (String):                     Index column name for the sparse categorical classes column. Default: 'CLASS'
"""
def csv_loader(path_data, path_imagedir, allowed_image_formats,
               training=True, ohe=True, ohe_range=None,
               col_sample="SAMPLE", col_class="CLASS"):
    # Load CSV file
    dt = pd.read_csv(path_data, sep=",", header=0)
    # Check if image index column exist and parse it
    if col_sample in dt.columns : index_list = dt[col_sample].tolist()
    else : raise Exception("Sample column (" + str(col_sample) + \
                           ") not available in CSV file!", path_data)
    # Ensure index list to contain strings
    index_list = [str(index) for index in index_list] 
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
    # Check if image ending is already in sample name by peaking first one
    if index_list[0].endswith("." + image_format) : image_format = None
    # Verify if all images are existing
    for sample in index_list:
        # Obtain image file path
        if image_format : img_file = sample + "." + image_format
        else : img_file = sample
        path_img = os.path.join(path_imagedir, img_file)
        # Check existance
        if not os.path.exists(path_img):
            raise Exception("Image does not exist / not accessible!",
                            'Sample: "' + sample + '"', path_img)

    # If CSV is for inference (no annotation data) -> return parsing
    if not training : return index_list, None, None, None, image_format

    # Try parsing with a sparse categorical class format (CSV Format 1)
    if not ohe:
        # Verify if provided classification column in in dataframe
        if col_class not in dt.columns:
            raise Exception("Provided classification column not in dataset!")
        # Obtain class information
        classes_sparse = dt[col_class].tolist()
        class_names = np.unique(classes_sparse).tolist()
        class_n = len(class_names)
        # Parse sparse categorical annotations to One-Hot Encoding
        class_ohe = pd.get_dummies(classes_sparse).to_numpy()
    # Try parsing one-hot encoded format (CSV Format 2)
    else:
        # Identify OHE columns
        if ohe_range is None : ohe_columns = dt.loc[:, dt.columns != col_sample]
        else : ohe_columns = dt.loc[:, ohe_range]
        # Parse information
        class_names = list(ohe_columns.columns)
        class_n = len(class_names)
        class_ohe = ohe_columns.to_numpy()

    # Validate if number of samples and number of annotations match
    if len(index_list) != len(class_ohe):
        raise Exception("Number of samples and annotation does not match!",
                        len(index_list), len(class_ohe))
    # Return parsed CSV data
    return index_list, class_ohe, class_n, class_names, image_format
