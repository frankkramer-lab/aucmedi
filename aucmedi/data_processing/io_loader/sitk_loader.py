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
import SimpleITK as sitk

#-----------------------------------------------------#
#              SITK Loader for AUCMEDI IO             #
#-----------------------------------------------------#
def sitk_loader(sample, path_imagedir, image_format=None, grayscale=True,
                resampling=(1.0, 1.0, 1.0), outside_value=0, **kwargs):
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
        from aucmedi.data_processing.io_loader import sitk_loader

        # Initialize input data reader
        ds = input_interface(interface="csv",
                            path_imagedir="dataset/nii_files/",
                            path_data="dataset/annotations.csv",
                            ohe=False, col_sample="ID", col_class="diagnosis")
        (samples, class_ohe, nclasses, class_names, image_format) = ds

        # Initialize DataGenerator with sitk_loader
        data_gen = DataGenerator(samples, "dataset/nii_files/", labels=class_ohe,
                                image_format=image_format, resize=None,
                                grayscale=True, resampling=(2.10, 1.48, 1.48),
                                loader=sitk_loader)
        ```

    Args:
        sample (str):               Sample name/index of an image.
        path_imagedir (str):        Path to the directory containing the images.
        image_format (str):         Image format to add at the end of the sample index for image loading.
        grayscale (bool):           Boolean, whether images are grayscale or RGB.
        resampling (tuple of float):Tuple of 3x floats with z,y,x mapping encoding voxel spacing.
                                    If passing `None`, no normalization will be performed.
        **kwargs (dict):            Additional parameters for the sample loader.
    """
    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    path_img = os.path.join(path_imagedir, img_file)
    # Load image via the SimpleITK package
    sample_itk = sitk.ReadImage(path_img)
    # Perform resampling
    if resampling is not None:
        # Extract information from sample
        shape = sample_itk.GetSize()
        spacing = sample_itk.GetSpacing()
        # Reverse resampling spacing to sITK mapping (z,y,x -> x,y,z)
        new_spacing = resampling[::-1]
        # Estimate output shape after resampling
        output_shape = []
        for t in zip(shape, spacing, new_spacing):
            s = int(t[0] * t[1] / t[2])
            output_shape.append(s)
        output_shape = tuple(output_shape)
        # Perform resampling via sITK
        sample_itk_resampled = sitk.Resample(sample_itk,
                                             output_shape,
                                             sitk.Transform(),
                                             sitk.sitkLinear,
                                             sample_itk.GetOrigin(),
                                             new_spacing,
                                             sample_itk.GetDirection(),
                                             outside_value,
                                             sitk.sitkFloat32)
    # Skip resampling if None
    else : sample_itk_resampled = sample_itk
    # Convert to NumPy
    img = sitk.GetArrayFromImage(sample_itk_resampled)
    # Add single channel axis
    if len(img.shape) == 3 : img = np.expand_dims(img, axis=-1)
    # Return image
    return img
