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
import pandas as pd


#-----------------------------------------------------#
#      Data Loader Interface based on Directories     #
#-----------------------------------------------------#
def directory_loader(path_imagedir, allowed_image_formats, training=True):
    """ Data Input Interface for loading a dataset in a directory-based structure.

    This **internal** function allows simple parsing of class annotations encoded in subdirectories.

    ???+ info "Input Formats"
        ```
        Format Directory - Training:
            - Class annotations are encoded via subdirectories
            - Images are provided in subdirectories

        Format Directory - Testing:
            - All images are provided in the directory
            - No class annotations
        ```

    **Expected structure for training:**
    ```
    images_dir/                     # path_imagedir = "dataset/images_dir"
        class_A/
            sample001.png
            sample002.png
            ...
            sample050.png
        class_B/                    # Directory / class names can be any String
            sample051.png           # like "diabetes", "cancer", ...
            sample052.png
            ...
            sample100.png
        ...
        class_C/
            sample101.png           # Sample names (indicies) should be unique!
            sample102.png
            ...
            sample150.png
    ```

    **Expected structure for testing:**
    ```
    images_dir/                     # path_imagedir = "dataset/images_dir"
        sample001.png
        sample002.png
        ...
        sample100.png
    ```

    Args:
        path_imagedir (str):                    Path to the directory containing the images or the subdirectories.
        allowed_image_formats (list of str):    List of allowed imaging formats. (provided by IO_Interface)
        training (bool):                        Boolean option whether annotation data is available.

    Returns:
        index_list (list of str):               List of sample/index encoded as Strings. Required in DataGenerator as
                                                `samples`.
        class_ohe (numpy.ndarray):              Classification list as One-Hot encoding. Required in DataGenerator as
                                                `labels`.
        class_n (int):                          Number of classes. Required in NeuralNetwork for Architecture design as
                                                `n_labels`.
        class_names (list of str):              List of names for corresponding classes. Used for later prediction
                                                storage or evaluation.
        image_format (str):                     Image format to add at the end of the sample index for image loading.
                                                Required in DataGenerator.
    """
    # Initialize some variables
    image_format = None
    index_list = []
    # Format - including class annotations encoded via subdirectories
    if training:
        class_names = []
        classes_sparse = []
        # Iterate over subdirectories
        for c, subdirectory in enumerate(sorted(os.listdir(path_imagedir))):
            # Skip items which are not a directory (metadata)
            if not os.path.isdir(os.path.join(path_imagedir, subdirectory)):
                continue
            class_names.append(subdirectory)
            # Iterate over each sample
            path_sd = os.path.join(path_imagedir, subdirectory)
            for file in sorted(os.listdir(path_sd)):
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
        for file in sorted(os.listdir(path_imagedir)):
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
