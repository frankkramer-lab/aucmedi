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
from PIL import Image
import json
import numpy as np
import pandas as pd

#-----------------------------------------------------#
#           I/O Data Interface based on CSV           #
#-----------------------------------------------------#
""" Data I/O Interface for loading a dataset via a CSV and an image directory.
    This function allow simple parsing of class annotations encoded in a CSV.

    CSV Format 1)
        - Name Column: "SAMPLE" -> String Value
        - Class Column: "CLASS" -> Sparse Categorical Classes (String/Integers)
        - Optional Meta Columns possible

    CSV Format 2)
        - Name Column: "SAMPLE"
        - One-Hot Encoded Class Columns:
            -> If OHE parameter provides column index values -> use these
            -> Else try to use all other columns as OHE columns
        - Optional Meta Columns only possible if OHE parameter provided

Arguments:
    path_csv (String):          Path to the csv file.
    path_imagedir (String):     Path to the directory containing the images.
    training (Boolean):         Boolean option whether annotation data is available.
    ohe (Integer list):         List of column index values if annotation encoded in OHE.
"""
def iointerface_csv(path_csv, path_imagedir, training=True, ohe=None):
    pass
    # check if CLASS column exist -> take this
    # else: OHE -> check if ohe integers are provided else all


# modi training
# - load from csv & single image dir
# - load from json & single image dir
# - load from directory with subdirectories according to classes

# modi testing
# load all images from a single directory


# list of indices
# list of class annotation
# Number of classes
# list of classification names




# how to handle binary/categorical/multi-label data?
