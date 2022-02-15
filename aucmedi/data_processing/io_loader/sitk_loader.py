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
# Internal libraries
from aucmedi.utils.resampling import Resampling

#-----------------------------------------------------#
#              SITK Loader for AUCMEDI IO             #
#-----------------------------------------------------#
""" SimpleITK Loader for loading of CT/MRI scans in NIfTI (nii) or Metafile (mha) format within the AUCMEDI pipeline.

    This loader is intended to load only 3D volumes with annotated voxel spacings.

    By default, volumes are normalized to voxel spacing 1.0 x 1.0 x 1.0.
    You can define a custom voxel spacing for the loader, by passing a aucmedi.utils.resampling.Resampling()
    class in the DataGenerator.


    Arguments:
        sample (String):                Sample name/index of an image.
        path_imagedir (String):         Path to the directory containing the images.
        image_format (String):          Image format to add at the end of the sample index for image loading.
        grayscale (Boolean):            Boolean, whether images are grayscale or RGB. (should always be True)
        resampling (Resampling class):  Passing of a Resampling class for normalizing voxel spacing.
        kwargs (Dictionary):            Additional parameters for the sample loader.
"""
def sitk_loader(sample, path_imagedir, image_format=None, grayscale=True,
                resampling=Resampling(), **kwargs):
    # Get image path
    if image_format : img_file = sample + "." + image_format
    else : img_file = sample
    path_img = os.path.join(path_imagedir, img_file)
    # Load image via the SimpleITK package
    sample_itk = sitk.ReadImage(path_img)
    # Extract voxel spacing
    spacing = sample_itk.GetSpacing()
    # Convert to NumPy
    img = sitk.GetArrayFromImage(sample_itk)
    # Transpose volume to be identical to sITK spacing encoding (x,y,z)
    img = np.transpose(img, axes=(2,1,0))
    # Add single channel axis
    if len(img.shape) == 3 : img = np.expand_dims(img, axis=-1)
    # Perform resampling
    img = resampling.transform(img, current_spacing=spacing)
    # Return image
    return img
