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
import numpy as np

#-----------------------------------------------------#
#             Cache Loader for AUCMEDI IO             #
#-----------------------------------------------------#
def cache_loader(sample, path_imagedir=None, image_format=None,
                 grayscale=False, two_dim=True, cache=None, **kwargs):
    """ Cache Loader for passing already loaded images within the AUCMEDI pipeline.

    The Cache Loader is an IO_loader function, which have to be passed to the 
    [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    The complete data management happens in the memory.
    Thus, for multiple images or common data set sizes, this is NOT recommended!

    !!! warning
        This functions requires to pass a Dictionary to the parameter `cache`!

    Dictionary structure: key=index as String; value=Image as NumPy array <br>
    e.g. cache = {"my_index_001": my_image}

    ???+ example
        ```python
        # Import required libraries
        from aucmedi import *
        from aucmedi.data_processing.io_loader import cache_loader

        # Encode information as dictionary
        cache = {"sample_a": image_a,
                 "sample_b": image_b,
                 "sample_n": image_n}

        # Obtain meta data
        my_labels = [[1,0], [0,1], [1,0]]      # one-hot encoded annotation matrix
        sample_list = cache.keys()

        # Initialize DataGenerator
        data_gen = DataGenerator(sample_list, None, labels=my_labels,
                                 resize=None, grayscale=False, two_dim=True,
                                 loader=cache_loader, cache=cache)
        ```

    Args:
        sample (str):               Sample name/index of an image.
        path_imagedir (str):        Path to the directory containing the images.
        image_format (str):         Image format to add at the end of the sample index for image loading.
        grayscale (bool):           Boolean, whether images are grayscale or RGB.
        two_dim (bool):             Boolean, whether image is 2D or 3D.
        cache (dict):               A Python dictionary containing one or multiple images.
        **kwargs (dict):            Additional parameters for the sample loader.
    """
    # Verify if a cache is provided
    if cache is None or type(cache) is not dict:
        raise TypeError("No dictionary was provided to cache_loader()!")
    # Obtain image from cache
    img = cache[sample]
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
