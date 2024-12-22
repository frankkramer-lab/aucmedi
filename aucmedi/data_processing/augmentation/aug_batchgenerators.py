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
# Third Party Libraries
import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (BrightnessMultiplicativeTransform,
                                                         ContrastAugmentationTransform, GammaTransform)
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform


#-----------------------------------------------------#
#        AUCMEDI Batchgenerators Augmentation         #
#-----------------------------------------------------#
class BatchgeneratorsAugmentation():
    """ The Batchgenerators Augmentation class performs diverse augmentation methods on given
        numpy array. The class acts as an easy to use function/interface for applying
        all types of augmentations with just one function call.

    The class can be configured beforehand by selecting desired augmentation techniques
    and method ranges or strength.
    Afterwards, the class is passed to the [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator]
    which utilizes it during batch generation.

    The specific configurations of selected methods can be adjusted by class variables.

    ???+ abstract "Build on top of the library"
        Batchgenerators from the DKFZ - https://github.com/MIC-DKFZ/batchgenerators

    ???+ abstract "Reference - Publication"
        Isensee Fabian, Jäger Paul, Wasserthal Jakob, Zimmerer David, Petersen Jens, Kohl Simon,
        Schock Justus, Klein Andre, Roß Tobias, Wirkert Sebastian, Neher Peter, Dinkelacker Stefan,
        Köhler Gregor, Maier-Hein Klaus (2020). batchgenerators - a python framework for data
        augmentation. doi:10.5281/zenodo.3632567
    """
    #-----------------------------------------------------#
    #              Augmentation Configuration             #
    #-----------------------------------------------------#
    # Define augmentation operator
    operator = None
    # Option for augmentation refinement (clipping)
    refine = True
    # Augmentation: Mirror
    aug_mirror = False
    aug_mirror_p = 0.5
    aug_mirror_axes = (0, 1, 2)
    # Augmentation: 90 degree rotate
    aug_rotate = False
    aug_rotate_p = 0.5
    aug_rotate_angleX = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    aug_rotate_angleY = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    aug_rotate_angleZ = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    # Augmentation: Brightness
    aug_brightness = False
    aug_brightness_p = 0.5
    aug_brightness_range = (0.5, 2)
    aug_brightness_per_channel = False
    # Augmentation: Contrast
    aug_contrast = False
    aug_contrast_p = 0.5
    aug_contrast_range = (0.3, 3.0)
    aug_contrast_per_channel = False
    aug_contrast_preserverange = True
    # Augmentation: Scale
    aug_scale = False
    aug_scale_p = 0.5
    aug_scale_range = (0.85, 1.25)
    # Augmentation: Gaussian Noise
    aug_gaussianNoise = False
    aug_gaussianNoise_p = 0.5
    aug_gaussianNoise_range = (0.0, 0.05)
    # Augmentation: Gamma
    aug_gamma = False
    aug_gamma_p = 0.5
    aug_gamma_range = (0.7, 1.5)
    aug_gamma_per_channel = False
    # Augmentation: Elastic Transformation
    aug_elasticTransform = False
    aug_elasticTransform_p = 0.5
    aug_elasticTransform_alpha = (0.0, 900.0)
    aug_elasticTransform_sigma = (9.0, 13.0)

    #-----------------------------------------------------#
    #                    Initialization                   #
    #-----------------------------------------------------#
    def __init__(self, image_shape, mirror=False, rotate=True, scale=True,
                 elastic_transform=False, gaussian_noise=True,
                 brightness=True, contrast=True, gamma=True):
        """ Initialization function for the Batchgenerators Augmentation interface.

        With boolean switches, it is possible to selected desired augmentation techniques.
        Recommended augmentation configurations are defined as class variables.
        Of course, these configs can be adjusted if needed.

        Args:
            image_shape (tuple of int):     Target shape of image, which will be passed to the neural network model.
            mirror (bool):                  Boolean, whether mirroring should be performed as data augmentation.
            rotate (bool):                  Boolean, whether rotations should be performed as data augmentation.
            scale (bool):                   Boolean, whether scaling should be performed as data augmentation.
            elastic_transform (bool):       Boolean, whether elastic deformation should be performed as data
                                            augmentation.
            gaussian_noise (bool):          Boolean, whether Gaussian noise should be added as data augmentation.
            brightness (bool):              Boolean, whether brightness changes should be added as data augmentation.
            contrast (bool):                Boolean, whether contrast changes should be added as data augmentation.
            gamma (bool):                   Boolean, whether gamma changes should be added as data augmentation.

        !!! warning
            If class variables (attributes) are modified, the internal augmentation operator
            has to be rebuild via the following call:

            ```python
            # initialize
            aug = BatchgeneratorsAugmentation(model.meta_input, mirror=True)

            # set probability to 100% = always
            aug.aug_mirror_p = 1.0
            # rebuild
            aug.build()
            ```

        Attributes:
            refine (bool):                  Boolean, whether clipping to [0,255] should be performed if outside of
                                            range.
            aug_mirror_p (float):           Probability of mirroring application if activated. Default=0.5.
            aug_rotate_p (float):           Probability of rotation application if activated. Default=0.5.
            aug_scale_p (float):            Probability of scaling application if activated. Default=0.5.
            aug_elasticTransform_p (float): Probability of elastic deformation application if activated. Default=0.5.
            aug_gaussianNoise_p (float):    Probability of Gaussian noise application if activated. Default=0.5.
            aug_brightness_p (float):       Probability of brightness application if activated. Default=0.5.
            aug_contrast_p (float):         Probability of contrast application if activated. Default=0.5.
            aug_gamma_p (float):            Probability of gamma application if activated. Default=0.5.
        """
        # Cache class variables
        self.image_shape = image_shape
        self.aug_mirror = mirror
        self.aug_rotate = rotate
        self.aug_scale = scale
        self.aug_elasticTransform = elastic_transform
        self.aug_gaussianNoise = gaussian_noise
        self.aug_brightness = brightness
        self.aug_contrast = contrast
        self.aug_gamma = gamma
        # Build augmentation operator
        self.build()

    #-----------------------------------------------------#
    #               Batchgenerators Builder               #
    #-----------------------------------------------------#
    def build(self):
        """ Builds the batchgenerators augmentator by initializing  all transformations.

        The activated transformation and their configurations are defined as
        class variables.

        -> Builds a new self.operator
        """
        # Initialize transform list
        transforms = []
        # Fill transform list
        if self.aug_mirror:
            tf = MirrorTransform(axes=self.aug_mirror_axes,
                                 p_per_sample=self.aug_mirror_p)
            transforms.append(tf)
        if self.aug_contrast:
            tf = ContrastAugmentationTransform(
                                 self.aug_contrast_range,
                                 preserve_range=self.aug_contrast_preserverange,
                                 per_channel=self.aug_contrast_per_channel,
                                 p_per_sample=self.aug_contrast_p)
            transforms.append(tf)
        if self.aug_brightness:
            tf = BrightnessMultiplicativeTransform(
                                 self.aug_brightness_range,
                                 per_channel=self.aug_brightness_per_channel,
                                 p_per_sample=self.aug_brightness_p)
            transforms.append(tf)
        if self.aug_gaussianNoise:
            tf = GaussianNoiseTransform(self.aug_gaussianNoise_range,
                                        p_per_sample=self.aug_gaussianNoise_p)
            transforms.append(tf)
        if self.aug_gamma:
            tf = GammaTransform(self.aug_gamma_range,
                                invert_image=False,
                                per_channel=self.aug_gamma_per_channel,
                                retain_stats=True,
                                p_per_sample=self.aug_gamma_p)
            transforms.append(tf)
        if self.aug_rotate or self.aug_scale or self.aug_elasticTransform:
            tf = SpatialTransform(self.image_shape,
                                  [i // 2 for i in self.image_shape],
                                  do_elastic_deform=self.aug_elasticTransform,
                                  alpha=self.aug_elasticTransform_alpha,
                                  sigma=self.aug_elasticTransform_sigma,
                                  do_rotation=self.aug_rotate,
                                  angle_x=self.aug_rotate_angleX,
                                  angle_y=self.aug_rotate_angleY,
                                  angle_z=self.aug_rotate_angleZ,
                                  do_scale=self.aug_scale,
                                  scale=self.aug_scale_range,
                                  border_mode_data='constant',
                                  border_cval_data=0,
                                  border_mode_seg='constant',
                                  border_cval_seg=0,
                                  order_data=3, order_seg=0,
                                  p_el_per_sample=self.aug_elasticTransform_p,
                                  p_rot_per_sample=self.aug_rotate_p,
                                  p_scale_per_sample=self.aug_scale_p,
                                  random_crop=False)
            transforms.append(tf)

        # Compose transforms
        self.operator = Compose(transforms)

    #-----------------------------------------------------#
    #                 Perform Augmentation                #
    #-----------------------------------------------------#
    def apply(self, image):
        """ Performs image augmentation with defined configuration on an image.

        This **internal** function is called in the DataGenerator during batch generation.

        Args:
            image (numpy.ndarray):          An image encoded as NumPy array with shape (z, y, x, channels).
        Returns:
            aug_image (numpy.ndarray):      An augmented / transformed image.
        """
        # Convert image to batchgenerators format (float32, channel first and with batch axis)
        image_bg = image.astype(np.float32)
        image_bg = np.expand_dims(image_bg, axis=0)
        image_bg = np.moveaxis(image_bg, -1, 1)
        # Perform image augmentation
        aug_image = self.operator(data=image_bg)["data"]
        # Remove batch axis and return to channel last
        aug_image = np.moveaxis(aug_image, 1, -1)
        aug_image = np.squeeze(aug_image, axis=0)
        # Perform clipping if image is out of grayscale/RGB encodings
        if self.refine and (np.min(aug_image) < 0 or np.max(aug_image) > 255):
            aug_image = np.clip(aug_image, a_min=0, a_max=255)
        # Return augmented image
        return aug_image
