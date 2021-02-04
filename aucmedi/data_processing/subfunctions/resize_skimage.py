#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
from skimage.transform import resize
from skimage import img_as_ubyte
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" DEPRECATED: Use aucmedi/data_processing/subfunctions/resize.py
    It uses albumentations which is way faster than skimage. (also bi-linear interpolation)
    Experiment: https://github.com/muellerdo/aucmedi/issues/13


    A Resize Subfunction class which resizes an images according to a desired shape.

    Shape have to be defined as tuple with x and y size:
    Resize(shape=(224, 224))

    Resizing is done via skimage resize transform which uses bi-linear interpolation.
    https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

Methods:
    __init__                Object creation function
    transform:              Resize an image input according to defined shape.
"""
class Resize(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, shape=(224, 224)):
        self.shape = shape

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Perform resizing into desired shape
        image_resized = resize(image, self.shape)
        # Transform image intensity values from [0,1] back to [0,255]
        image_resized = img_as_ubyte(image_resized)
        # Return resized image
        return image_resized
