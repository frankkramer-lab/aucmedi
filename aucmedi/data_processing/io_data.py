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
# Internal libraries
import aucmedi.data_processing.io_interfaces as io

#-----------------------------------------------------#
#                   Static Variables                  #
#-----------------------------------------------------#
ACCEPTABLE_IMAGE_FORMATS = ["jpeg", "jpg", "tif", "tiff", "png", "bmp", "gif"]

#-----------------------------------------------------#
#            IO Interface Class for AUCMEDI           #
#-----------------------------------------------------#
""" Data I/O Interface for all kinds of dataset structures.
    Different image file structures and annotation information are processed by
    corresponding format interfaces.
    Basically a wrapper class for calling the correct format interface.

Methods:
    __init__                IO Interface creation and configuration.
    load_dataset:           Loading a dataset from disk with associated interface.
    save_inference:         Save a prediction to disk with associated interface.
"""
class IO_Interface:
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function which acts as a configuration port.

    Possible format interfaces: ["csv", "json", "directory"]

    Arguments:
        path_imagedir (String):         Path to the directory containing the images.
        interface (String):             String defining format interface for loading/storing data.
        path_data (String):             Path to the index/class annotation file if required. (csv/json)
        n_classes (Integer):            Number of classes in the dataset.
        training (Boolean):             Boolean option whether annotation data is available.
        ohe (Boolean):                  Boolean option whether annotation data is sparse categorical or one-hot encoded.
        kwargs (Dictionary):            Save a prediction to disk with associated interface.
    """
    def __init__(self, interface, path_imagedir, path_data=None,
                 n_classes=None, training=True, ohe=False, **kwargs):
        # Cache class variable
        self.path_imagedir = path_imagedir
        self.path_data = path_data
        self.interface = interface.lower()
        self.n_classes = n_classes
        self.training = training
        self.ohe = ohe
        self.config = kwargs
        # Verify if provided interface is valid
        if interface not in ["csv", "json", "dictionary"]:
            raise Exception("Unknown interface code provided.", interface)
        # Verify that annotation file is available if CSV/JSON interface is used
        if self.interface in ["csv", "json"] and path_data is None:
            raise Exception("No annoation file provided for CSV/JSON interface!")

    #---------------------------------------------#
    #               Dataset Loading               #
    #---------------------------------------------#
    """ Function for loading a dataset with associated format interface.     """
    def load_dataset(self):
        # Initialize parameter dictionary
        parameters = {"path_data": self.path_data,
                      "path_imagedir": self.path_imagedir,
                      "allowed_image_formats": ACCEPTABLE_IMAGE_FORMATS,
                      "training": self.training, "ohe": self.ohe}
        # Identify correct dataset loader and parameters for CSV
        if self.interface == "csv":
            ds_loader = io.csv_loader
            additional_parameters = ["path_data", "ohe", "ohe_range",
                                     "col_sample", "col_class"]
            for para in additional_parameters:
                if para in self.config : parameters[para] = self.config[para]
        # Identify correct dataset loader and parameters for JSON
        elif self.interface == "json":
            pass
        # Identify correct dataset loader and parameters for dictionary
        elif self.interface == "dictionary":
            pass

        # Run dataset loading
        ds = ds_loader(**parameters)
        (index_list, class_ohe, nc_io, class_names, image_format) = ds

        return ds

# io class
# communication interface to data generator

# modi training
# - load from csv & single image dir
# - load from json & single image dir
# - load from directory with subdirectories according to classes

# modi testing
# load all images from a single directory

# handles io for data loading and inference
