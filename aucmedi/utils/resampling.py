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
# Internal libraries/scripts
import aucmedi.data_processing.augmentation.volumentations.augmentations.functional import resize

#-----------------------------------------------------#
#              Utility Class: Resampling              #
#-----------------------------------------------------#
""" A Resampling class which resizes an images according to a desired voxel spacing.

    Be aware that this is not a Subfunction and can not be passed to a Subfunction list of a DataGenerator!
    (due to it requires passing the image voxel spacing in Resampling.transform() as parameter as well)

    Resizing is done via volumentations resize transform which uses bi-linear interpolation by default.
    https://github.com/frankkramer-lab/aucmedi/tree/master/aucmedi/data_processing/augmentation/volumentations

Methods:
    __init__                Object creation function
    transform:              Resample an image input according to defined voxel spacing.
"""
class Resampling():
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, spacing=(1.0, 1.0, 1.0), interpolation=1):
        # Cache parameter
        self.spacing = np.array(spacing)
        self.interpolation = interpolation

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image, current_spacing):
        # Calculate spacing ratio
        ratio = np.array(current_spacing) / self.spacing
        # Calculate new shape
        new_shape = tuple(np.floor(image.shape[0:-1] * ratio).astype(int))
        # Perform resizing into desired shape
        image_resampled = resize(image, new_shape,
                                 interpolation=self.interpolation)
        # Return resized image
        return image_resampled
