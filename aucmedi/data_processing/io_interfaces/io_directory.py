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
#      Data Loader Interface based on Directories     #
#-----------------------------------------------------#
""" Data Input Interface for loading a dataset via a CSV and an image directory.
    This function allow simple parsing of class annotations encoded in a CSV.

    Format Directory - Training:
        - Class annotations are encoded via subdirectories
        - Images are provided in subdirectories

    Format Directory - Testing:
        - All images are provided in the directory
        - No class annotations

Arguments:
    path_imagedir (String):                 Path to the directory containing the images or the subdirectories.
    allowed_image_formats (String list):    List of allowed imaging formats. (provided by IO_Interface)
    training (Boolean):                     Boolean option whether annotation data is available.
"""
def directory_loader(path_imagedir, allowed_image_formats, training=True):
    # Initialize some variables
    image_format = None
    index_list = []
    # Format - including class annotations encoded via subdirectories
    if training:
        class_names = []
        classes_sparse = []
        # Iterate over subdirectories
        for c, subdirectory in enumerate(os.listdir(path_imagedir)):
            class_names.append(subdirectory)
            # Iterate over each sample
            for file in os.listdir(os.path.join(path_imagedir, subdirectory)):
                sample = os.path.join(subdirectory, file)
                index_list.append(sample)
                classes_sparse.append(c)
        # Parse sparse categorical annotations to One-Hot Encoding
        class_n = len(class_names)
        class_ohe = pd.get_dummies(classes_sparse).to_numpy()
        # Return parsing
        return index_list, class_ohe, class_n, class_names, image_format
    # Format - excluding class annotations -> only testing images
    else:
        # Iterate over all images
        for file in os.listdir(path_imagedir):
            # Identify image format by peaking first image
            if image_format is None:
                format = file.split(".")[-1]
                if format.lower() in allowed_image_formats or \
                   format.upper() in allowed_image_formats:
                   image_format = format
            # Add sample to list
            index_list.append(file[:-(len(format)+1)])
        # Raise Exception if image format is unknown
        if image_format is None:
            raise Exception("Unknown image format.", path_imagedir)
        # Return parsing
        return index_list, None, None, None, image_format
