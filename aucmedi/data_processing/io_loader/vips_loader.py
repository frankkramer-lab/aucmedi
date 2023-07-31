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
import re
import numpy as np
import pyvips

SUPPORTED_OPS = ["crop"]

#-----------------------------------------------------#
#              VIPS Loader for AUCMEDI IO             #
#-----------------------------------------------------#
#TODO edit info text
def vips_loader(sample, path_imagedir, image_format=None, grayscale=True,
                **kwargs):
    """ SimpleITK Loader for loading of CT/MRI scans in NIfTI (nii) or Metafile (mha) format within the AUCMEDI pipeline.

    The SimpleITK Loader is an IO_loader function, which have to be passed to the
    [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    This loader is intended to load only 3D volumes with annotated voxel spacings.

    By default, volumes are normalized to voxel spacing 1.0 x 1.0 x 1.0. <br>
    You can define a custom voxel spacing for the loader, by passing a tuple of spacings as parameter 'resampling'. <br>

    ???+ info
        The SimpleITK Loader utilizes SimpleITK for sample loading: <br>
        https://simpleitk.readthedocs.io/en/master/IO.html

    ???+ example
        ```python
        # Import required libraries
        from aucmedi import *
        from aucmedi.data_processing.io_loader import vips_loader

        # Initialize input data reader
        ds = input_interface(interface="csv",
                            path_imagedir="dataset/",
                            path_data="dataset/annotations.csv",
                            ohe=False, col_sample="ID", col_class="diagnosis",
                            load_params=["x", "y", "height", "width"])
        (samples, class_ohe, nclasses, class_names, image_format) = ds

        # Initialize DataGenerator with vips_loader
        data_gen = DataGenerator(samples, "dataset/", labels=class_ohe,
                                image_format=image_format, resize=None,
                                grayscale=True, loader=vips_loader,
                                crop=["x", "y", "height", "width"])
        ```

    Args:
        sample (str):               Sample name/index of an image.
        path_imagedir (str):        Path to the directory containing the images.
        image_format (str):         Image format to add at the end of the sample index for image loading.
        grayscale (bool):           Boolean, whether images are grayscale or RGB.
        crop (list):                Load parameter keys that are parsed to Offset X, Y; Extend X, Y Default: None
        **kwargs (dict):            Additional parameters for the sample loader.
    """
    #parse all ops from params
    op_lst = {}
    op_params = []
    for op in SUPPORTED_OPS:
        if op in kwargs.keys():
            op_lst[op] = kwargs[op]
            op_params.extend(kwargs[op])
    #parse parameters from file
    pattern = r'[^;=\[]+?=[^;=\]]+'
    param_map = {}
    for match in re.finditer(pattern, sample[sample.find("["):sample.rfind("]")]):
        s = match.group(0)
        eq = s.find("=")
        param_map[s[:eq]] = s[eq + 1:]
    sample = sample[:sample.find("[")] #remove params
    #Check if parameters are all here
    if not all([param in param_map.keys() for param in op_params]):
        raise Exception("Not all expected parameters have been parsed.")

    #resolve parameters
    op_lst = {k: [param_map[v1] for v1 in v] for k, v in op_lst.items()}
    param_map = {k: v for k, v in param_map.items() if not k in op_params}

    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    #reconstruct load parameters
    if (len(param_map) > 0):
        img_file += "[" + ";".join([k + "=" + v for k, v in param_map.items()]) + "]"

    path_img = os.path.join(path_imagedir, img_file)
    
    # Load image via the VIPS package
    sample_vips = pyvips.Image.new_from_file(path_img)
    page_height = sample_vips.get("page-height")
    #Apply Operations
    if (page_height * 3 == sample_vips.height): #rolled format
        pages = [sample_vips.crop(0, y, sample_vips.width, page_height)
         for y in range(0, sample_vips.height, page_height)]
        # join pages band-wise to make an interleaved image
        sample_vips = pages[0].bandjoin(pages[1:])

        # set the rgb hint
        sample_vips = sample_vips.copy(interpretation="srgb")

    if ("crop" in op_lst.keys()):
        params = op_lst["crop"]
        sample_vips = sample_vips.crop(params[0], params[1], params[2], params[3])

    # Convert to NumPy
    img = sample_vips.numpy()
    # Add single channel axis
    if len(img.shape) == 3 : img = np.expand_dims(img, axis=-1)
    # Return image
    return img
