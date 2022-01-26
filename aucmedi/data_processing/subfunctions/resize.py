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
from albumentations import Compose
from albumentations import Resize as aug_resize
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" A Resize Subfunction class which resizes an images according to a desired shape.

    Shape have to be defined as tuple with x and y size:
    Resize(shape=(224, 224))

    Resizing is done via albumentations resize transform which uses bi-linear interpolation by default.
    https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/

Methods:
    __init__                Object creation function
    transform:              Resize an image input according to defined shape.
"""
class Resize(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, shape=(224, 224)):
        # Initialize resizing transform
        self.shape = shape
        self.aug_transform = Compose([aug_resize(width=shape[1],
                                                 height=shape[0],
                                                 p=1.0, always_apply=True)])

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform resizing into desired shape
        image_resized = self.aug_transform(image=image)["image"]
        # Return resized image
        return image_resized
