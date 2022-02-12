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
import albumentations as alb
import aucmedi.data_processing.augmentation.volumentations as vol
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#             Subfunction class: Cropping             #
#-----------------------------------------------------#
""" A Crop Subfunction class which randomly crops a desired shape from an image.

2D image: Shape have to be defined as tuple with x and y size:
    Crop(shape=(224, 224))

    Cropping is done via albumentations RandomCrop transform.
    https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop

3D volume: Shape have to be defined as tuple with x, y and z size:
    Crop(shape=(224, 224, 244))

    Cropping is done via volumentations RandomCrop transform.
    https://github.com/frankkramer-lab/aucmedi/tree/master/aucmedi/data_processing/augmentation/volumentations

Methods:
    __init__                Object creation function
    transform:              Crops an image input with the defined shape.
"""
class Crop(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, shape=(224, 224)):
        # If 2D -> Initialize albumentations RandomCrop
        if len(shape) == 2:
            self.aug_transform = alb.Compose([alb.RandomCrop(
                                                     height=shape[0],
                                                     width=shape[1],
                                                     p=1.0,
                                                     always_apply=True)])
        # If 3D -> Initialize volumentations RandomCrop
        else:
            self.aug_transform = vol.Compose([vol.RandomCrop(
                                                     shape,
                                                     p=1.0,
                                                     always_apply=True)])
        # Cache shape
        self.shape = shape

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform cropping on image
        image_cropped = self.aug_transform(image=image)["image"]
        # Return cropped image
        return image_cropped
