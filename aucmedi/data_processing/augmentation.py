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
from albumentations import Compose
import albumentations.augmentations as ai

#-----------------------------------------------------#
#              AUCMEDI Image Augmentation             #
#-----------------------------------------------------#
""" The Image Augmentation class performs diverse augmentation methods on given
    numpy array. The class acts as an easy to use function/interface for applying
    all types of augmentations with just one function call.

    The class can be configured beforehand by selecting desired augmentation techniques
    and method ranges or strength.
    Afterwards, the class is passed to the Data Generator which utilizes it during
    batch generation.

    The specific configurations of selected methods can be adjusted by class variables.

    Build on top of the library: Albumentations
    https://github.com/albumentations-team/albumentations
"""
class Image_Augmentation():
    #-----------------------------------------------------#
    #              Augmentation Configuration             #
    #-----------------------------------------------------#
    # Define augmentation operator
    operator = None
    # Augmentation: Flip
    aug_flip = False
    aug_flip_p = 0.5
    # Augmentation: 90 degree rotate
    aug_rotate = False
    aug_rotate_p = 0.5
    # Augmentation: Brightness
    aug_brightness = False
    aug_brightness_p = 0.5
    aug_brightness_limits = (-0.1, 0.1)
    # Augmentation: Contrast
    aug_contrast = False
    aug_contrast_p = 0.5
    aug_contrast_limits = (-0.1, 0.1)
    # Augmentation: Saturation shift
    aug_saturation = False
    aug_saturation_p = 0.5
    aug_saturation_limits = 0.1
    # Augmentation: Hue shift
    aug_hue = False
    aug_hue_p = 0.5
    aug_hue_limits = (-0.1, 0.1)
    # Augmentation: Scale
    aug_scale = False
    aug_scale_p = 0.5
    aug_scale_limits = (0.9, 1.1)
    # Augmentation: Crop
    aug_crop = False
    aug_crop_p = 0.5
    aug_crop_shape = (244, 244)
    # Augmentation: Grid Distortion
    aug_gridDistortion = False
    aug_gridDistortion_p = 0.5
    # Augmentation: Image Compression (JPEG)
    aug_compression = False
    aug_compression_p = 0.5
    aug_compression_limits = (90, 100)
    # Augmentation: Gaussian Noise
    aug_gaussianNoise = False
    aug_gaussianNoise_p = 0.5
    # Augmentation: Gaussian Blur
    aug_gaussianBlur = False
    aug_gaussianBlur_p = 0.5
    # Augmentation: Downscale
    aug_downscaling = False
    aug_downscaling_p = 0.5
    aug_downscaling_effect = 0.10
    # Augmentation: Gamma
    aug_gamma = False
    aug_gamma_p = 0.5
    aug_gamma_limit = (90, 110)
    # Augmentation: Elastic Transformation
    aug_elasticTransform = False
    aug_elasticTransform_p = 0.5

    #-----------------------------------------------------#
    #                    Initialization                   #
    #-----------------------------------------------------#
    """ Initialization function for the Image Augmentation interface.

        With boolean switches, it is possible to selected desired augmentation techniques.
        Recommended augmentation configurations are defined as class variables.
        Of course, these configs can be adjusted if needed.
    """
    def __init__(self, flip=True, rotate=True, brightness=True, contrast=True,
                 saturation=True, hue=True, scale=True, crop=False,
                 grid_distortion=False, compression=False, gaussian_noise=False,
                 gaussian_blur=False, downscaling=False, gamma=False,
                 elastic_transform=False):
        # Cache class variables
        self.aug_flip = flip
        self.aug_rotate = rotate
        self.aug_brightness = brightness
        self.aug_contrast = contrast
        self.aug_scale = scale
        self.aug_crop = crop
        self.aug_saturation = saturation
        self.aug_hue = hue
        self.aug_compression = compression
        self.aug_gaussianNoise = gaussian_noise
        self.aug_gaussianBlur= gaussian_blur
        self.aug_downscaling = downscaling
        self.aug_gamma = gamma
        self.aug_gridDistortion = grid_distortion
        self.aug_elasticTransform = elastic_transform
        # Build augmentation operator
        self.build()

    #-----------------------------------------------------#
    #                Albumentations Builder               #
    #-----------------------------------------------------#
    """ Builds the albumenations augmentator by initializing  all transformations.
        The activated transformation and their configurations are defined as
        class variables.

        -> Builds a new self.operator
    """
    def build(self):
        # Initialize transform list
        transforms = []
        # Fill transform list
        if self.aug_flip:
            tf = ai.Flip(p=self.aug_flip_p)
            transforms.append(tf)
        if self.aug_rotate:
            tf = ai.RandomRotate90(p=self.aug_rotate_p)
            transforms.append(tf)
        if self.aug_brightness:
            tf = ai.RandomBrightnessContrast(brightness_limit=self.aug_brightness_limits,
                                             contrast_limit=0,
                                             p=self.aug_brightness_p)
            transforms.append(tf)
        if self.aug_contrast:
            tf = ai.RandomBrightnessContrast(contrast_limit=self.aug_contrast_limits,
                                             brightness_limit=0,
                                             p=self.aug_contrast_p)
            transforms.append(tf)
        if self.aug_saturation:
            tf = ai.ColorJitter(brightness=0, contrast=0, hue=0,
                                saturation=self.aug_saturation_limits,
                                p=self.aug_saturation_p)
            transforms.append(tf)
        if self.aug_hue:
            tf = ai.ColorJitter(brightness=0, contrast=0, saturation=0,
                                hue=self.aug_hue_limits,
                                p=self.aug_hue_p)
            transforms.append(tf)
        if self.aug_scale:
            tf = ai.RandomScale(scale_limit=self.aug_scale_limits,
                                p=self.aug_scale_p)
            transforms.append(tf)
        if self.aug_crop:
            tf = ai.RandomCrop(width=self.aug_crop_shape[0],
                               height=self.aug_crop_shape[1],
                               p=self.aug_crop_p)
            transforms.append(tf)
        if self.aug_gridDistortion:
            tf = ai.GridDistortion(p=self.aug_gridDistortion_p)
            transforms.append(tf)
        if self.aug_compression:
            tf = ai.ImageCompression(quality_lower=self.aug_compression_limits[0],
                                     quality_upper=self.aug_compression_limits[1],
                                     p=self.aug_compression_p)
            transforms.append(tf)
        if self.aug_gaussianNoise:
            tf = ai.GaussNoise(p=self.aug_gaussianNoise_p)
            transforms.append(tf)
        if self.aug_gaussianBlur:
            tf = ai.GlassBlur(p=self.aug_gaussianBlur_p)
            transforms.append(tf)
        if self.aug_downscaling:
            tf = ai.Downscale(scale_min=self.aug_downscaling_effect,
                              scale_max=self.aug_downscaling_effect,
                              p=self.aug_downscaling_p)
            transforms.append(tf)
        if self.aug_gamma:
            tf = ai.RandomGamma(gamma_limit=self.aug_gamma_limit,
                                p=self.aug_gamma_p)
            transforms.append(tf)
        if self.aug_elasticTransform:
            tf = ai.ElasticTransform(p=self.aug_elasticTransform_p)
            transforms.append(tf)

        # Compose transforms
        self.operator = Compose(transforms)

    #-----------------------------------------------------#
    #                 Perform Augmentation                #
    #-----------------------------------------------------#
    """ Performs image augmentation with defined configuration on an image.
        This function is called in the Data Generator during batch generation.

        Arguments:
            - image (NumPy array):      An image encoded as NumPy array with shape (x, y, channels).
        Returns:
            - aug_image (NumPy array):  An augmented / transformed image.
    """
    def apply(self, image):
        # Perform image augmentation
        aug_image = self.operator(image=image)["image"]
        # Return augmented image
        return aug_image
