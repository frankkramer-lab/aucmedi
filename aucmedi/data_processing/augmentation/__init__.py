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
#                    Documentation                    #
#-----------------------------------------------------#
""" The Augmentation classes of AUCMEDI allow creating interfaces to powerful
    augmentation frameworks and easily integrate them into the AUCMEDI pipeline.

An Augmentation class is a preprocessing method, which is randomly applied on each sample
if provided to a [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

???+ warning
    Augmentation should only be applied to a **training** DataGenerator!

    For test-time augmentation, [aucmedi.ensemble.augmenting][] should be used.

Data augmentation is a technique that can be used to artificially expand the size
of a training dataset by creating modified versions of images in the dataset.

The point of data augmentation is, that the model will learn meaningful patterns
instead of meaningless characteristics due to a small data set size.

???+ info "Data Augmentation Interfaces"
    | Interface                                                                                | Description                                                           |
    | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
    | [ImageAugmentation][aucmedi.data_processing.augmentation.aug_image]                     | Interface to package: Albumentations. Handles only images (2D data).  |
    | [VolumeAugmentation][aucmedi.data_processing.augmentation.aug_volume]                   | Interface to package: Volumentations. Handles only volumes (3D data). |
    | [BatchgeneratorsAugmentation][aucmedi.data_processing.augmentation.aug_batchgenerators] | Interface to package: batchgenerators (DKFZ). Handles images and volumes (2D+3D data). |

**Recommendation:** <br>
- For images (2D data): ImageAugmentation() <br>
- For volumes (3D data): BatchgeneratorsAugmentation() <br>

???+ example
    **For 2D data:**
    ```python
    from aucmedi import *

    aug = ImageAugmentation(flip=True, rotate=True, brightness=True, contrast=True,
                 saturation=True, hue=True, scale=True, crop=False,
                 grid_distortion=False, compression=False, gaussian_noise=False,
                 gaussian_blur=False, downscaling=False, gamma=False,
                 elastic_transform=False)

    datagen = DataGenerator(samples=index_list,
                            path_imagedir="dataset/images/",
                            labels=class_ohe,
                            data_aug=aug,
                            resize=model.meta_input,
                            image_format=image_format)
    ```

    **For 3D data:**
    ```python
    from aucmedi import *

    aug = BatchgeneratorsAugmentation(model.meta_input, mirror=False, rotate=True,
                 scale=True, elastic_transform=False, gaussian_noise=True,
                 brightness=True, contrast=True, gamma=True)

    datagen = DataGenerator(samples=index_list,
                            path_imagedir="dataset/volumes/",
                            labels=class_ohe,
                            data_aug=aug,
                            resize=model.meta_input,
                            image_format=image_format)
    ```
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.data_processing.augmentation.aug_image import ImageAugmentation
from aucmedi.data_processing.augmentation.aug_volume import VolumeAugmentation
from aucmedi.data_processing.augmentation.aug_batchgenerators import BatchgeneratorsAugmentation
