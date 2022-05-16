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
import numpy as np
# Internal libraries
from aucmedi import Image_Augmentation, Volume_Augmentation, DataGenerator
from aucmedi.ensemble.aggregate import aggregate_dict
from aucmedi.data_processing.io_loader import image_loader

#-----------------------------------------------------#
#       Ensemble Learning: Inference Augmenting       #
#-----------------------------------------------------#
def predict_augmenting(model, prediction_generator, n_cycles=10, aggregate="mean"):
    """ Inference Augmenting function for automatically augmenting unknown images for prediction.

    The predictions of the augmented images are aggregated together via the provided aggregate function.

    ???+ example
        ```python
        # Import libraries
        from aucmedi.ensemble import predict_augmenting
        from aucmedi import Image_Augmentation, DataGenerator

        # Initialize testing DataGenerator with desired Data Augmentation
        test_aug = Image_Augmentation(flip=True, rotate=True, brightness=False, contrast=False))
        test_gen = DataGenerator(samples_test, "images_dir/",
                                 data_aug=test_aug,
                                 resize=model.meta_input,
                                 standardize_mode=model.meta_standardize)

        # Compute predictions via Augmenting
        preds = predict_augmenting(model, test_gen, n_cycles=15, aggregate="majority_vote")
        ```

    The aggregate function can be either self initialized with an AUCMEDI aggregate function
    or a custom made aggregate function, or by calling an AUCMEDI aggregate function by name.

    !!! info
        Possible aggregate function names: ["mean", "median", "majority_vote", "softmax"]

        More about aggregate functions can be found here: [aggregate][aucmedi.ensemble.aggregate]

    The Data Augmentation class instance from the DataGenerator will be used for inference augmenting,
    can be either predefined or leaving `None`. If the `data_aug` is `None`, a Data Augmentation class
    instance is automatically created which applies rotation and flipping augmentations.

    ???+ warning
        The passed DataGenerator will be re-initialized!
        This can result into redundant image preparation if `prepare_images=True`.

    Args:
        model (Neural_Network):                 Instance of a AUCMEDI neural network class.
        prediction_generator (DataGenerator):   A data generator which will be used for Augmenting based inference.
        n_cycles (int):                         Number of image augmentations, which should be created per sample.
        aggregate (str or aggregate Function):  Aggregate function class instance or a string for an AUCMEDI aggregate function.
    """
    # Initialize aggregate function if required
    if isinstance(aggregate, str) and aggregate in aggregate_dict:
        agg_fun = aggregate_dict[aggregate]()
    else : agg_fun = aggregate

    # Initialize image augmentation if none provided (only flip, rotate)
    if prediction_generator.data_aug is None and len(model.input_shape) == 3:
        data_aug = Image_Augmentation(flip=True, rotate=True, scale=False,
                                     brightness=False, contrast=False,
                                     saturation=False, hue=False, crop=False,
                                     grid_distortion=False, compression=False,
                                     gamma=False, gaussian_noise=False,
                                     gaussian_blur=False, downscaling=False,
                                     elastic_transform=False)
    elif prediction_generator.data_aug is None and len(model.input_shape) == 4:
        data_aug = Volume_Augmentation(flip=True, rotate=True, scale=False,
                                      brightness=False, contrast=False,
                                      saturation=False, hue=False, crop=False,
                                      grid_distortion=False, compression=False,
                                      gamma=False, gaussian_noise=False,
                                      gaussian_blur=False, downscaling=False,
                                      elastic_transform=False)
    else : data_aug = prediction_generator.data_aug
    # Multiply sample list for prediction according to number of cycles
    samples_aug = np.repeat(prediction_generator.samples, n_cycles)

    # Re-initialize DataGenerator for inference
    aug_gen = DataGenerator(samples_aug,
                            path_imagedir=prediction_generator.path_imagedir,
                            labels=None,
                            batch_size=prediction_generator.batch_size,
                            data_aug=data_aug,
                            seed=prediction_generator.seed,
                            subfunctions=prediction_generator.subfunctions,
                            shuffle=False,
                            standardize_mode=prediction_generator.standardize_mode,
                            resize=prediction_generator.resize,
                            grayscale=prediction_generator.grayscale,
                            prepare_images=prediction_generator.prepare_images,
                            sample_weights=None,
                            image_format=prediction_generator.image_format,
                            loader=prediction_generator.sample_loader,
                            workers=prediction_generator.workers,
                            **prediction_generator.kwargs)

    # Compute predictions with provided model
    preds_all = model.predict(aug_gen)

    # Ensemble inferences via aggregate function
    preds_ensembled = []
    for i in range(0, len(prediction_generator.samples)):
        # Identify subset for a single sample
        j = i*n_cycles
        subset = preds_all[j:j+n_cycles]
        # Aggregate predictions
        pred_sample = agg_fun.aggregate(subset)
        # Add prediction to prediction list
        preds_ensembled.append(pred_sample)
    # Convert prediction list to NumPy
    preds_ensembled = np.asarray(preds_ensembled)

    # Return ensembled predictions
    return preds_ensembled
