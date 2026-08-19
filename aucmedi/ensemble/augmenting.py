#==============================================================================#
#  Author:       Fabian Wehr                                                   #
#  Copyright:    2026 IT-Infrastructure for Translational Medical Research,    #
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
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import numpy as np
from torch.utils.data import DataLoader

# Internal libraries
from aucmedi import ImageAugmentation, VolumeAugmentation, create_batch_loader

from aucmedi.ensemble.aggregate import aggregate_dict
from aucmedi.data_processing.io_loader import image_loader


# -----------------------------------------------------#
#       Ensemble Learning: Inference Augmenting       #
# -----------------------------------------------------#
def predict_augmenting(model, prediction_generator, n_cycles=10, aggregate="mean"):
    """Inference Augmenting function for automatically augmenting unknown images for prediction.

    The predictions of the augmented images are aggregated via the provided Aggregate function.

    ???+ example
        ```python
        # Import libraries
        from aucmedi.ensemble import predict_augmenting
        from aucmedi import ImageAugmentation, create_batch_loader

        # Initialize testing DataLoader with desired Data Augmentation
        test_aug = ImageAugmentation(flip=True, rotate=True, brightness=False, contrast=False)
        test_loader = create_batch_loader(samples_test, "images_dir/",
                                          data_aug=test_aug,
                                          resize=model.arch_resolution,
                                          standardize_mode=model.arch_standardize)

        # Compute predictions via Augmenting
        preds = predict_augmenting(model, test_loader, n_cycles=15, aggregate="majority_vote")
        ```

    The inclusion of the Aggregate function can be achieved in multiple ways:

    - self-initialization with an AUCMEDI Aggregate function,
    - use a string key to call an AUCMEDI Aggregate function by name, or
    - implementing a custom Aggregate function by extending the [AUCMEDI base class for Aggregate functions][aucmedi.ensemble.aggregate.agg_base]

    !!! info
        Description and list of implemented Aggregate functions can be found here:
        [Aggregate][aucmedi.ensemble.aggregate]

    The Data Augmentation class instance from the BatchLoader will be used for inference augmenting.
    It can either be predefined or remain `None`. If the `data_aug` is `None`, a Data Augmentation class
    instance is automatically created which applies rotation and flipping augmentations.

    ???+ warning
        The passed generator will be re-initialized!
        This can result in redundant image preparation if `prepare_images=True`.

    ??? reference "Reference for Ensemble Learning Techniques"
        Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2022).
        An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks.
        arXiv e-print: [https://arxiv.org/abs/2201.11440](https://arxiv.org/abs/2201.11440)

    Args:
        model (NeuralNetwork):                  Instance of an AUCMEDI neural network class.
        prediction_generator (DataLoader):      A generator which will be used for Augmenting based inference.
        n_cycles (int):                         Number of augmented copies to generate per sample.
        aggregate (str or Aggregate):           Aggregate function class instance or a string for an AUCMEDI Aggregate function.

    Returns:
        preds (numpy.ndarray):                  A NumPy array of ensembled predictions with shape (n_samples, n_labels).
    """
    # Initialize aggregate function if required
    if isinstance(aggregate, str) and aggregate in aggregate_dict:
        agg_fun = aggregate_dict[aggregate]()
    else:
        agg_fun = aggregate

    # Unwrap DataLoader to access the underlying Dataset/BatchGenerator attributes
    num_workers = 0
    batch_size = None
    shuffle = False
    if isinstance(prediction_generator, DataLoader):
        num_workers = getattr(prediction_generator, "num_workers", 0)
        batch_size = prediction_generator.batch_size
        shuffle = getattr(prediction_generator, "shuffle", False)
        # TODO: capture sampler
        prediction_generator = prediction_generator.dataset

    # Initialize image augmentation if none provided (only flip, rotate)
    if prediction_generator.data_aug is None and len(model.input_shape) == 3:
        data_aug = ImageAugmentation(
            flip=True,
            rotate=True,
            scale=False,
            brightness=False,
            contrast=False,
            saturation=False,
            hue=False,
            crop=False,
            grid_distortion=False,
            compression=False,
            gamma=False,
            gaussian_noise=False,
            gaussian_blur=False,
            downscaling=False,
            elastic_transform=False,
        )
    elif prediction_generator.data_aug is None and len(model.input_shape) == 4:
        data_aug = VolumeAugmentation(
            flip=True,
            rotate=True,
            scale=False,
            brightness=False,
            contrast=False,
            saturation=False,
            hue=False,
            crop=False,
            grid_distortion=False,
            compression=False,
            gamma=False,
            gaussian_noise=False,
            gaussian_blur=False,
            downscaling=False,
            elastic_transform=False,
        )
    else:
        data_aug = prediction_generator.data_aug
    # Multiply sample list for prediction according to number of cycles
    samples_aug = np.repeat(prediction_generator.samples, n_cycles)

    # Assume prediction_generator is a BatchGenerator or DataGenerator (Pytorch Dataset) and use its attributes directly
    loader_args = prediction_generator.__dict__
    num_workers = getattr(prediction_generator, "num_workers", num_workers)
    batch_size = getattr(prediction_generator, "batch_size", batch_size)
    shuffle = getattr(prediction_generator, "shuffle", shuffle)
    loader_args.setdefault("num_workers", num_workers)
    loader_args.setdefault("batch_size", batch_size)

    # Re-initialize BatchLoader for inference
    aug_loader = create_batch_loader(
        samples_aug,
        path_imagedir=loader_args.get("path_imagedir"),
        labels=None,
        metadata=loader_args.get("metadata"),
        batch_size=loader_args.get("batch_size"),
        data_aug=data_aug,
        seed=loader_args.get("seed"),
        subfunctions=loader_args.get("subfunctions"),
        shuffle=False,
        standardize_mode=loader_args.get("standardize_mode"),
        resize=loader_args.get("resize"),
        grayscale=loader_args.get("grayscale"),
        prepare_images=loader_args.get("prepare_images"),
        sample_weights=None,
        image_format=loader_args.get("image_format"),
        loader=loader_args.get("sample_loader"),
        two_dim=loader_args.get("two_dim"),
        num_workers=num_workers,
        **prediction_generator.kwargs
    )

    # Compute predictions with provided model
    preds_all = model.predict(aug_loader)

    # Ensemble inferences via aggregate function
    preds_ensembled = []
    for i in range(0, len(prediction_generator.samples)):
        # Identify subset for a single sample
        j = i * n_cycles
        subset = preds_all[j : j + n_cycles]
        # Aggregate predictions
        pred_sample = agg_fun.aggregate(subset)
        # Add prediction to prediction list
        preds_ensembled.append(pred_sample)
    # Convert prediction list to NumPy
    preds_ensembled = np.asarray(preds_ensembled)

    # Return ensembled predictions
    return preds_ensembled
