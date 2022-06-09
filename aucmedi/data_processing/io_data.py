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
""" List of accepted image formats. """

#-----------------------------------------------------#
#             Input Interface for AUCMEDI             #
#-----------------------------------------------------#
def input_interface(interface, path_imagedir, path_data=None, training=True,
                    ohe=False, image_format=None, **kwargs):
    """ Data Input Interface for all automatically extract various information of dataset structures.

    Different image file structures and annotation information are processed by
    corresponding format interfaces. These extracted information can be parsed to the
    [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator] and the
    [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork].

    The input_interface() function is the first of the three pillars of AUCMEDI.

    ??? info "Pillars of AUCMEDI"
        - [aucmedi.data_processing.io_data.input_interface][]
        - [aucmedi.data_processing.data_generator.DataGenerator][]
        - [aucmedi.neural_network.model.NeuralNetwork][]

    Basically a wrapper function for calling the correct format interface,
    which loads a dataset from disk via the associated format parser.

    Possible format interfaces: `["csv", "json", "directory"]`

    ???+ info "Format Interfaces"
        | Interface      | Internal Function                                                    | Description                                  |
        | -------------- | -------------------------------------------------------------------- | -------------------------------------------- |
        |  `"csv"`       | [io_csv()][aucmedi.data_processing.io_interfaces.io_csv]             | Storing class annotations in a CSV file.     |
        |  `"directory"` | [io_directory()][aucmedi.data_processing.io_interfaces.io_directory] | Storing class annotations in subdirectories. |
        |  `"json"`      | [io_json()][aucmedi.data_processing.io_interfaces.io_json]           | Storing class annotations in a JSON file.    |

    ???+ example
        ```python
        # AUCMEDI library
        from aucmedi import *

        # Initialize input data reader
        ds = input_interface(interface="csv",                       # Interface type
                             path_imagedir="dataset/images/",
                             path_data="dataset/annotations.csv",
                             ohe=False, col_sample="ID", col_class="diagnosis")
        (index_list, class_ohe, nclasses, class_names, image_format) = ds

        # Pass variables to other AUCMEDI pillars like DataGenerator
        datagen = DataGenerator(samples=index_list,                 # from input_interface()
                                path_imagedir="dataset/images/",
                                labels=class_ohe,                   # from input_interface()
                                image_format=image_format)          # from input_interface()
        ```

    Args:
        path_imagedir (str):            Path to the directory containing the images.
        interface (str):                String defining format interface for loading/storing data.
        path_data (str):                Path to the index/class annotation file if required. (csv/json)
        training (bool):                Boolean option whether annotation data is available.
        ohe (bool):                     Boolean option whether annotation data is sparse categorical or one-hot encoded.
        image_format (str):             Force to use a specific image format. By default, image format is determined automatically.
        **kwargs (dict):                Additional parameters for the format interfaces.

    Returns:
        index_list (list of str):       List of sample/index encoded as Strings. Required in DataGenerator as `samples`.
        class_ohe (numpy.ndarray):      Classification list as One-Hot encoding. Required in DataGenerator as `labels`.
        class_n (int):                  Number of classes. Required in NeuralNetwork for Architecture design as `n_labels`.
        class_names (list of str):      List of names for corresponding classes. Used for later prediction storage or evaluation.
        image_format (str):             Image format to add at the end of the sample index for image loading. Required in DataGenerator.
    """
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
        raise Exception("No annotation file provided for CSV/JSON interface!")

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
            if para in kwargs : parameters[para] = kwargs[para]
    # Identify correct dataset loader and parameters for JSON format
    elif interface == "json" : ds_loader = io.json_loader
    # Identify correct dataset loader and parameters for directory format
    elif interface == "directory":
        ds_loader = io.directory_loader
        del parameters["ohe"]
        del parameters["path_data"]

    # Load the dataset with the selected format interface and return results
    return ds_loader(**parameters)
