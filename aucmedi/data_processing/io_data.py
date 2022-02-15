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
# Internal libraries
import aucmedi.data_processing.io_interfaces as io

#-----------------------------------------------------#
#                   Static Variables                  #
#-----------------------------------------------------#
ACCEPTABLE_IMAGE_FORMATS = ["jpeg", "jpg", "tif", "tiff", "png", "bmp", "gif",
                            "npy", "nii", "gz", "mha"]

#-----------------------------------------------------#
#             Input Interface for AUCMEDI             #
#-----------------------------------------------------#
""" Data Input Interface for all kinds of dataset structures.
    Different image file structures and annotation information are processed by
    corresponding format interfaces.
    Basically a wrapper function for calling the correct format interface,
    which loads a dataset from disk via the associated format parser.

    Possible format interfaces: ["csv", "json", "directory"]

    Arguments:
        path_imagedir (String):         Path to the directory containing the images.
        interface (String):             String defining format interface for loading/storing data.
        path_data (String):             Path to the index/class annotation file if required. (csv/json)
        training (Boolean):             Boolean option whether annotation data is available.
        ohe (Boolean):                  Boolean option whether annotation data is sparse categorical or one-hot encoded.
        image_format (String):          Force to use a specific image format. By default, image format is determined automatically.
        kwargs (Dictionary):            Additional parameters for the format interfaces.
"""
def input_interface(interface, path_imagedir, path_data=None, training=True,
                    ohe=False, image_format=None, **config):
    # Transform selected interface to lower case
    interface = interface.lower()
    # Pass image format if provided
    if image_format != None : allowed_image_formats = [image_format]
    else : allowed_image_formats = ACCEPTABLE_IMAGE_FORMATS
    # Verify if provided interface is valid
    if interface not in ["csv", "json", "directory"]:
        raise Exception("Unknown interface code provided.", interface)
    # Verify that annotation file is available if CSV/JSON interface is used
    if interface in ["csv", "json"] and path_data is None:
        raise Exception("No annoation file provided for CSV/JSON interface!")

    # Initialize parameter dictionary
    parameters = {"path_data": path_data,
                  "path_imagedir": path_imagedir,
                  "allowed_image_formats": allowed_image_formats,
                  "training": training, "ohe": ohe}
    # Identify correct dataset loader and parameters for CSV format
    if interface == "csv":
        ds_loader = io.csv_loader
        additional_parameters = ["ohe_range", "col_sample", "col_class"]
        for para in additional_parameters:
            if para in config : parameters[para] = config[para]
    # Identify correct dataset loader and parameters for JSON format
    elif interface == "json" : ds_loader = io.json_loader
    # Identify correct dataset loader and parameters for directory format
    elif interface == "directory":
        ds_loader = io.directory_loader
        del parameters["ohe"]
        del parameters["path_data"]

    # Load the dataset with the selected format interface and return results
    return ds_loader(**parameters)
