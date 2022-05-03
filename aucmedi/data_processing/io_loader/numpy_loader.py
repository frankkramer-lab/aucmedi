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

#-----------------------------------------------------#
#             Numpy Loader for AUCMEDI IO             #
#-----------------------------------------------------#
def numpy_loader(sample, path_imagedir, image_format=None, grayscale=False,
                 two_dim=True, **kwargs):
    """ NumPy Loader for image loading within the AUCMEDI pipeline.

    The NumPy Loader is an IO_loader function, which have to be passed to the
    [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    The NumPy load function `np.load(path_img, allow_pickle=True)` is used.

    ???+ example
        ```python
        # Import required libraries
        from aucmedi import *
        from aucmedi.data_processing.io_loader import numpy_loader

        # Initialize input data reader
        ds = input_interface(interface="csv",
                             path_imagedir="dataset/npy_files/",
                             path_data="dataset/annotations.csv",
                             ohe=False, col_sample="ID", col_class="diagnosis")
        (samples, class_ohe, nclasses, class_names, image_format) = ds

        # Initialize DataGenerator with numpy_loader
        data_gen = DataGenerator(samples, "dataset/npy_files/", labels=class_ohe,
                                 image_format=image_format, resize=None,
                                 grayscale=True, two_dim=False,
                                 loader=numpy_loader)
        ```

    Args:
        sample (str):               Sample name/index of an image.
        path_imagedir (str):        Path to the directory containing the images.
        image_format (str):         Image format to add at the end of the sample index for image loading.
        grayscale (bool):           Boolean, whether images are grayscale or RGB.
        two_dim (bool):             Boolean, whether image is 2D or 3D.
        **kwargs (dict):            Additional parameters for the sample loader.
    """
    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    path_img = os.path.join(path_imagedir, img_file)
    # Load image via the NumPy package
    img = np.load(path_img, allow_pickle=True)
    # Verify image shape for grayscale & 2D
    if grayscale and two_dim:
        # Add channel axis and return image
        if len(img.shape) == 2:
            return np.reshape(img, img.shape + (1,))
        # Just return image
        elif len(img.shape) == 3 and img.shape[-1] == 1:
            return img
        # Throw Exception
        else:
            raise ValueError("Parameter 2D & Grayscale: Expected either 2D " + \
                             "without channel axis or 3D with single channel" + \
                             " axis, but got:", img.shape, len(img.shape))
    # Verify image shape for grayscale & 3D
    elif grayscale and not two_dim:
        # Add channel axis and return image
        if len(img.shape) == 3:
            return np.reshape(img, img.shape + (1,))
        # Just return image
        elif len(img.shape) == 4 and img.shape[-1] == 1:
            return img
        # Throw Exception
        else:
            raise ValueError("Parameter 3D & Grayscale: Expected either 3D " + \
                             "without channel axis or 4D with single channel" + \
                             " axis, but got:", img.shape, len(img.shape))
    # Verify image shape for rgb & 2D
    elif not grayscale and two_dim:
        # Just return image
        if len(img.shape) == 3 and img.shape[-1] == 3:
            return img
        # Throw Exception
        else:
            raise ValueError("Parameter 2D & RGB: Expected 3D array " + \
                             "including a single channel axis, but got:",
                             img.shape, len(img.shape))
    # Verify image shape for rgb & 3D
    elif not grayscale and not two_dim:
        # Just return image
        if len(img.shape) == 4 and img.shape[-1] == 3:
            return img
        # Throw Exception
        else:
            raise ValueError("Parameter 3D & RGB: Expected 4D array " + \
                             "including a single channel axis, but got:",
                             img.shape, len(img.shape))
