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
import numpy as np
# Internal libraries
from aucmedi import Image_Augmentation, DataGenerator
from aucmedi.ensembler.aggregate import *

#-----------------------------------------------------#
#       Ensemble Learning: Inference Augmenting       #
#-----------------------------------------------------#
""" sadsad

    aggregate function bla

    img aug none -> automatically created

Arguments:
    model (Neural_Network):         Instance of a AUCMEDI neural network class.
    samples (List of Strings):      List of sample/index encoded as Strings.
    path_imagedir (String):         Path to the directory containing the images.
    n_cycles (Integer):             Number of image augmentations, which should be created per sample.
    img_aug (ImageAugmentation):    Image Augmentation class instance which performs diverse data augmentation techniques.
    aggregate (String or aggregate Function):
                                    Aggregate function class instance or a string for an AUCMEDI aggregate function.
    image_format (String):          Image format to add at the end of the sample index for image loading.
    batch_size (Integer):           Number of samples inside a single batch.
    resize (Tuple of Integers):     Resizing shape consisting of a X and Y size.
    grayscale (Boolean):            Boolean, whether images are grayscale or RGB.
    subfunctions (List of Subfunctions):
                                    List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
    standardize_mode (String):      Standardization modus in which image intensity values are scaled.
    seed (Integer):                 Seed to ensure reproducibility for random function.
    workers (Integer):              Number of workers. If n_workers > 1 = use multi-threading for image preprocessing.
"""
def predict_augmenting(model, samples, path_imagedir, n_cycles=10, img_aug=None,
                       aggregate="mean", image_format=None, batch_size=32,
                       resize=(224, 224), grayscale=False, subfunctions=[],
                       standardize_mode="tf", seed=None, workers=1):
    # Initialize image augmentation if none provided (only flip, rotate and scalign)
    if img_aug is None:
        img_aug = Image_Augmentation(flip=True, rotate=True, scale=True,
                                     brightness=False, contrast=False,
                                     saturation=False, hue=False, crop=False,
                                     grid_distortion=True, compression=False,
                                     gamma=False, gaussian_noise=False,
                                     gaussian_blur=False, downscaling=False,
                                     elastic_transform=False)
    # Multiply sample list for prediction according to number of cycles
    samples_aug = np.repeat(samples, n_cycles)

    # Create DataGenerator for inference
    aug_gen = DataGenerator(samples_aug, path_imagedir, labels=None,
                            batch_size=batch_size, img_aug=img_aug, seed=seed,
                            subfunctions=subfunctions, shuffle=False,
                            standardize_mode=standardize_mode, resize=resize,
                            grayscale=grayscale, prepare_images=False,
                            sample_weights=None, image_format=image_format,
                            workers=workers)

    # Compute predictions with provided model
    preds_all = model.predict(aug_gen)

    # Ensemble inferences via aggregate function

    # -> take subsets of prediction table with n_cycles size
    # -> pass each subset to aggregate
    # -> aggregate has to compute <mean> over rows
    # -> aggregate return single row prediction
    # -> combine single row predictions to original prediction format (list of list in NumPy)

    # return preds
