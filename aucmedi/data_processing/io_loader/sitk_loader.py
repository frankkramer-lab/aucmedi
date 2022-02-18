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
""" SimpleITK Loader for loading of CT/MRI scans in NIfTI (nii) or Metafile (mha) format within the AUCMEDI pipeline.

    This loader is intended to load only 3D volumes with annotated voxel spacings.

    By default, volumes are normalized to voxel spacing 1.0 x 1.0 x 1.0.
    You can define a custom voxel spacing for the loader, by passing a tuple of spacings as parameter 'resampling'.

    Arguments:
        sample (String):                Sample name/index of an image.
        path_imagedir (String):         Path to the directory containing the images.
        image_format (String):          Image format to add at the end of the sample index for image loading.
        grayscale (Boolean):            Boolean, whether images are grayscale or RGB. (should always be True)
        resampling (Tuple):             Tuple of 3x floats with z,y,x mapping.
        kwargs (Dictionary):            Additional parameters for the sample loader.
"""
def sitk_loader(sample, path_imagedir, image_format=None, grayscale=True,
                resampling=(1.0, 1.0, 1.0), outside_value=0, **kwargs):
    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    path_img = os.path.join(path_imagedir, img_file)
    # Load image via the SimpleITK package
    sample_itk = sitk.ReadImage(path_img)
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
    # Convert to NumPy
    img = sitk.GetArrayFromImage(sample_itk_resampled)
    # Add single channel axis
    if len(img.shape) == 3 : img = np.expand_dims(img, axis=-1)
    # Return image
    return img
