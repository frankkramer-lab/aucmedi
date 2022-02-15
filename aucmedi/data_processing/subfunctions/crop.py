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
import albumentations
import aucmedi.data_processing.augmentation.volumentations as volumentations
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#             Subfunction class: Cropping             #
#-----------------------------------------------------#
""" A Crop Subfunction class which center/randomly crops a desired shape from an image.

    List of valid modes for parameter "mode": ["center", "random"]

2D image: Shape have to be defined as tuple with x and y size:
    Crop(shape=(224, 224))

    Cropping is done via albumentations CenterCrop and RandomCrop transform.
    https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CenterCrop
    https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop

3D volume: Shape have to be defined as tuple with x, y and z size:
    Crop(shape=(224, 224, 244))

    Cropping is done via volumentations CenterCrop and RandomCrop transform.
    https://github.com/frankkramer-lab/aucmedi/tree/master/aucmedi/data_processing/augmentation/volumentations

Methods:
    __init__                Object creation function
    transform:              Crops an image input with the defined shape.
"""
class Crop(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, shape=(224, 224), mode="center"):
        # Initialize parameter
        params = {"p":1.0, "always_apply":True}
        # Select augmentation module and add further parameter depending on dimension
        if len(shape) == 2:
            params["height"] = shape[0]
            params["width"] = shape[1]
            mod = albumentations
        elif len(shape) == 3:
            params["shape"] = shape
            mod = volumentations
        else : raise ValueError("Shape for cropping has to be 2D or 3D!", shape)
        # Initialize cropping transform
        if mode == "center":
            self.aug_transform = mod.Compose([mod.CenterCrop(**params)])
        elif mode == "random":
            self.aug_transform = mod.Compose([mod.RandomCrop(**params)])
        else : raise ValueError("Unknown mode for crop Subfunction", mode,
                                "Possibles modes are: ['center', 'random']")
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
