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
#                    Documentation                    #
#-----------------------------------------------------#
""" The IO Loader functions of AUCMEDI allow loading samples from datasets in different file formats.

These functions are called **internally** via the [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

!!! info "IO_loader Functions"
    | Interface                                                        | Description                                  |
    | ---------------------------------------------------------------- | -------------------------------------------- |
    | [image_loader()][aucmedi.data_processing.io_loader.image_loader] | Image Loader for image loading via Pillow. |
    | [sitk_loader()][aucmedi.data_processing.io_loader.sitk_loader]   | SimpleITK Loader for loading NIfTI (nii) or Metafile (mha) formats.    |
    | [numpy_loader()][aucmedi.data_processing.io_loader.numpy_loader] | NumPy Loader for image loading of .npy files.    |
    | [cache_loader()][aucmedi.data_processing.io_loader.cache_loader] | Cache Loader for passing already loaded images. |

    Parameters defined in `**kwargs` are passed down to IO_loader functions.

???+ example
    ```python
    # Import required libraries
    from aucmedi import *

    # Initialize input data reader
    ds = input_interface(interface="csv",
                         path_imagedir="dataset/images/",
                         path_data="dataset/annotations.csv",
                         ohe=False, col_sample="ID", col_class="diagnosis")
    (samples, class_ohe, nclasses, class_names, image_format) = ds

    # Initialize DataGenerator with by default using image_loader
    data_gen = DataGenerator(samples, "dataset/images/", labels=class_ohe,
                             image_format=image_format, resize=None)

    # Initialize DataGenerator with manually selected image_loader
    from aucmedi.data_processing.io_loader import image_loader
    data_gen = DataGenerator(samples, "dataset/images/", labels=class_ohe,
                             image_format=image_format, resize=None,
                             loader=image_loader)
    ```
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.data_processing.io_loader.image_loader import image_loader
from aucmedi.data_processing.io_loader.numpy_loader import numpy_loader
from aucmedi.data_processing.io_loader.sitk_loader import sitk_loader
from aucmedi.data_processing.io_loader.cache_loader import cache_loader
