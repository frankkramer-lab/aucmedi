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
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" A Resize Subfunction class which resizes an images according to a desired shape.

2D image: Shape have to be defined as tuple with x and y size:
    Resize(shape=(224, 224))

    Resizing is done via albumentations resize transform which uses bi-linear interpolation by default.
    https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/

3D volume: Shape have to be defined as tuple with x, y and z size:
    Resize(shape=(128, 128, 128))

    Resizing is done via volumentations resize transform which uses bi-linear interpolation by default.
    https://github.com/frankkramer-lab/aucmedi/tree/master/aucmedi/data_processing/augmentation/volumentations

Methods:
    __init__                Object creation function
    transform:              Resize an image input according to defined shape.
"""
class Resize(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, shape=(224, 224), interpolation=1):
        # If 2D -> Initialize albumentations Resize
        if len(shape) == 2:
            self.aug_transform = alb.Compose([alb.Resize(height=shape[0],
                                                        width=shape[1],
                                                        p=1.0,
                                                        always_apply=True)])
        # If 3D -> Initialize volumentations Resize
        else:
            self.aug_transform = vol.Compose([vol.Resize(shape,
                                                         p=1.0,
                                                         always_apply=True)])
        # Cache shape
        self.shape = shape

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform resizing into desired shape
        image_resized = self.aug_transform(image=image)["image"]
        # Return resized image
        return image_resized
