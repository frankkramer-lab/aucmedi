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
from PIL import Image

#-----------------------------------------------------#
#             Image Loader for AUCMEDI IO             #
#-----------------------------------------------------#
def image_loader(sample, path_imagedir, image_format=None, grayscale=False,
                 **kwargs):
    """ Image Loader for image loading within the AUCMEDI pipeline.

    The Image Loader is an IO_loader function, which have to be passed to the
    [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    ???+ info
        The Image Loader utilizes Pillow for image loading: <br>
        https://github.com/python-pillow/Pillow

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

    Args:
        sample (str):               Sample name/index of an image.
        path_imagedir (str):        Path to the directory containing the images.
        image_format (str):         Image format to add at the end of the sample index for image loading.
        grayscale (bool):           Boolean, whether images are grayscale or RGB.
        **kwargs (dict):            Additional parameters for the sample loader.
    """
    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    path_img = os.path.join(path_imagedir, img_file)
    # Load image via the PIL package
    img_raw = Image.open(path_img)
    # Convert image to grayscale or rgb
    if grayscale : img_converted = img_raw.convert('LA')
    else : img_converted = img_raw.convert('RGB')
    # Convert image to NumPy
    img = np.asarray(img_converted)
    # Perform additional preprocessing if grayscale image
    if grayscale:
        # Remove maximum value and keep only intensity
        img = img[:,:,0]
        # Reshape image to create a single channel
        img = np.reshape(img, img.shape + (1,))
    # Return image
    return img
