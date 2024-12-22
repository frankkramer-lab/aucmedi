#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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
# Python Standard Library
import random
import warnings

# Third Party Libraries
import numpy as np
from volumentations import Compose
from volumentations import augmentations as ai


#-----------------------------------------------------#
#             AUCMEDI Volume Augmentation             #
#-----------------------------------------------------#
class VolumeAugmentation():
    """ The Volume Augmentation class performs diverse augmentation methods on given
        numpy array. The class acts as an easy to use function/interface for applying
        all types of augmentations with just one function call.

    The class can be configured beforehand by selecting desired augmentation techniques
    and method ranges or strength.
    Afterwards, the class is passed to the [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator]
    which utilizes it during batch generation.

    The specific configurations of selected methods can be adjusted by class variables.

    ???+ abstract "Build on top of the library"
        Volumentations based on albumentations.

        ```
        volumentations: Originally developed by @ashawkey and @ZFTurbo.
                        Initially inspired by albumentations library for augmentation of 2D images.
                        https://github.com/muellerdo/volumentations
                        https://github.com/ZFTurbo/volumentations
                        https://github.com/ashawkey/volumentations
        albumentations: https://github.com/albumentations-team/albumentations
        ```

        The Volumentations package was further continued by us to ensure ongoing development and support.

        For more details, please read the README under:
        https://github.com/muellerdo/volumentations
    """
    #-----------------------------------------------------#
    #              Augmentation Configuration             #
    #-----------------------------------------------------#
    # Define augmentation operator
    operator = None
    # Option for augmentation refinement (padding, cropping and clipping)
    refine = True
    # Augmentation: Flip
    aug_flip = False
    aug_flip_p = 0.5
    # Augmentation: 90 degree rotate
    aug_rotate = False
    aug_rotate_p = 0.5
    # Augmentation: Brightness
    aug_brightness = False
    aug_brightness_p = 0.5
    aug_brightness_limits = 0.1
    # Augmentation: Contrast
    aug_contrast = False
    aug_contrast_p = 0.5
    aug_contrast_limits = 0.1
    # Augmentation: Saturation shift
    aug_saturation = False
    aug_saturation_p = 0.5
    aug_saturation_limits = 0.1
    # Augmentation: Hue shift
    aug_hue = False
    aug_hue_p = 0.5
    aug_hue_limits = 0.1
    # Augmentation: Scale
    aug_scale = False
    aug_scale_p = 0.5
    aug_scale_limits = (0.9, 1.1)
    # Augmentation: Crop
    aug_crop = False
    aug_crop_p = 0.5
    aug_crop_shape = (64, 64, 64)
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
    aug_downscaling_effect = 0.25
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
    def __init__(self, flip=True, rotate=True, brightness=True, contrast=True,
                 saturation=True, hue=True, scale=True, crop=False,
                 grid_distortion=False, compression=False, gaussian_noise=False,
                 gaussian_blur=False, downscaling=False, gamma=False,
                 elastic_transform=False):
        """ Initialization function for the Volume Augmentation interface.

        With boolean switches, it is possible to selected desired augmentation techniques.
        Recommended augmentation configurations are defined as class variables.
        Of course, these configs can be adjusted if needed.

        Args:
            flip (bool):                    Boolean, whether flipping should be performed as data augmentation.
            rotate (bool):                  Boolean, whether rotations should be performed as data augmentation.
            brightness (bool):              Boolean, whether brightness changes should be added as data augmentation.
            contrast (bool):                Boolean, whether contrast changes should be added as data augmentation.
            saturation (bool):              Boolean, whether saturation changes should be added as data augmentation.
            hue (bool):                     Boolean, whether hue changes should be added as data augmentation.
            scale (bool):                   Boolean, whether scaling should be performed as data augmentation.
            crop (bool):                    Boolean, whether scaling cropping be performed as data augmentation.
            grid_distortion (bool):         Boolean, whether grid_distortion should be performed as data augmentation.
            compression (bool):             Boolean, whether compression should be performed as data augmentation.
            gaussian_noise (bool):          Boolean, whether gaussian noise should be added as data augmentation.
            gaussian_blur (bool):           Boolean, whether gaussian blur should be added as data augmentation.
            downscaling (bool):             Boolean, whether downscaling should be added as data augmentation.
            gamma (bool):                   Boolean, whether gamma changes should be added as data augmentation.
            elastic_transform (bool):       Boolean, whether elastic deformation should be performed as data
                                            augmentation.

        !!! warning
            If class variables (attributes) are modified, the internal augmentation operator
            has to be rebuild via the following call:

            ```python
            # initialize
            aug = VolumeAugmentation(flip=True)

            # set probability to 100% = always
            aug.aug_flip_p = 1.0
            # rebuild
            aug.build()
            ```

        Attributes:
            refine (bool):                  Boolean, whether clipping to [0,255] and padding/cropping should be
                                            performed if outside of range.
            aug_flip_p (float):             Probability of flipping application if activated. Default=0.5.
            aug_rotate_p (float):           Probability of rotation application if activated. Default=0.5.
            aug_brightness_p (float):       Probability of brightness application if activated. Default=0.5.
            aug_contrast_p (float):         Probability of contrast application if activated. Default=0.5.
            aug_saturation_p (float):       Probability of saturation application if activated. Default=0.5.
            aug_hue_p (float):              Probability of hue application if activated. Default=0.5.
            aug_scale_p (float):            Probability of scaling application if activated. Default=0.5.
            aug_crop_p (float):             Probability of crop application if activated. Default=0.5.
            aug_grid_distortion_p (float):  Probability of grid_distortion application if activated. Default=0.5.
            aug_compression_p (float):      Probability of compression application if activated. Default=0.5.
            aug_gaussianNoise_p (float):    Probability of gaussian noise application if activated. Default=0.5.
            aug_gaussianBlur_p (float):     Probability of gaussian blur application if activated. Default=0.5.
            aug_downscaling_p (float):      Probability of downscaling application if activated. Default=0.5.
            aug_gamma_p (float):            Probability of gamma application if activated. Default=0.5.
            aug_elasticTransform_p (float): Probability of elastic deformation application if activated. Default=0.5.
        """
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
        self.aug_gaussianBlur = gaussian_blur
        self.aug_downscaling = downscaling
        self.aug_gamma = gamma
        self.aug_gridDistortion = grid_distortion
        self.aug_elasticTransform = elastic_transform
        # Build augmentation operator
        self.build()

    #-----------------------------------------------------#
    #                Albumentations Builder               #
    #-----------------------------------------------------#
    def build(self):
        """ Builds the albumenations augmentator by initializing  all transformations.

        The activated transformation and their configurations are defined as
        class variables.

        -> Builds a new self.operator
        """
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
            tf = ai.ColorJitter(brightness=self.aug_brightness_limits,
                                contrast=0, hue=0, saturation=0,
                                p=self.aug_brightness_p)
            transforms.append(tf)
        if self.aug_contrast:
            tf = ai.ColorJitter(brightness=0, contrast=self.aug_contrast_limits,
                                hue=0, saturation=0,
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
            tf = ai.RandomCrop(self.aug_crop_shape,
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
            tf = ai.GaussianNoise(p=self.aug_gaussianNoise_p)
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

    This **internal** function is called in the DataGenerator during batch generation.

    Args:
        image (numpy.ndarray):          An image encoded as NumPy array with shape (z, y, x, channels).
    Returns:
        aug_image (numpy.ndarray):      An augmented / transformed image.
    """
    def apply(self, image):
        # Verify that image is in grayscale/RGB encoding
        if np.min(image) < 0 or np.max(image) > 255:
            warnings.warn("Image Augmentation: A value of the image is lower than 0 or higher than 255.",
                          "Volumentations expects images to be in grayscale/RGB!",
                          np.min(image), np.max(image))
        # Cache image shape
        org_shape = image.shape
        # Perform image augmentation
        aug_image = self.operator(image=image)["image"]
        # Perform padding & cropping if image shape changed
        if self.refine and aug_image.shape != org_shape:
            aug_image = ai.pad(aug_image, new_shape=org_shape)
            offset = (random.random(), random.random(), random.random())
            aug_image = ai.random_crop(aug_image,
                                       org_shape[0], org_shape[1], org_shape[2],
                                       offset[0], offset[1], offset[2])
        # Perform clipping if image is out of grayscale/RGB encodings
        if self.refine and (np.min(aug_image) < 0 or np.max(aug_image) > 255):
            aug_image = np.clip(aug_image, a_min=0, a_max=255)
        # Return augmented image
        return aug_image
