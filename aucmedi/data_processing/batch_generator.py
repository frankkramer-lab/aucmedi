# ==============================================================================#
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from torch.utils.data import Dataset
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import repeat
import tempfile
import pickle
import os

# Internal libraries
from aucmedi.data_processing.io_loader import image_loader
from aucmedi.data_processing.subfunctions import Standardize, Resize


# -----------------------------------------------------#
#                 Torch Data Generator                #
# -----------------------------------------------------#
class BatchGenerator(Dataset):
    """PyTorch Dataset that pre-forms full batches for AUCMEDI model training and inference.

    `BatchGenerator` is the second pillar of AUCMEDI. It handles image loading,
    subfunction application, resizing, data augmentation, and standardization,
    returning a ready-to-use batch tensor from each `__getitem__` call.

    ??? warning "Non-standard Dataset contract"
        Unlike a standard PyTorch `Dataset`, `__getitem__` returns a **full pre-formed
        batch** (shape `(batch_size, C, H, W)`), not a single sample. It must therefore
        be wrapped in a
        [WrapperLoader][aucmedi.data_processing.wrapper_loader.WrapperLoader]
        with `batch_size=None` to prevent PyTorch's `DataLoader` from re-batching.

        Use [create_batch_loader][aucmedi.data_processing.wrapper_loader.create_batch_loader]
        as the standard entry point — it constructs the `BatchGenerator` and the
        `WrapperLoader` together.

    ??? info "Pillars of AUCMEDI"
        - [aucmedi.data_processing.io_data.input_interface][]
        - [aucmedi.data_processing.batch_generator.BatchGenerator][]
        - [aucmedi.neural_network.model.NeuralNetwork][]

    ???+ example
        ```python
        from aucmedi import *
        from aucmedi.data_processing.wrapper_loader import create_batch_loader

        model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.ResNet50")

        train_loader = create_batch_loader(
            samples=samples[:100],
            path_imagedir="images_dir/",
            labels=class_ohe[:100],
            image_format=image_format,
            resize=model.arch_resolution,
            standardize_mode=model.arch_standardize,
        )
        model.train(train_loader, epochs=50)

        test_loader = create_batch_loader(
            samples=samples[100:150],
            path_imagedir="images_dir/",
            image_format=image_format,
            resize=model.arch_resolution,
            standardize_mode=model.arch_standardize,
        )
        preds = model.predict(test_loader)
        ```

    The batch pipeline runs in this order:

    1. Image loading (via `loader` / IO-loader function)
    2. Application of `subfunctions` (sequentially)
    3. Resize to `resize`
    4. Data augmentation (`data_aug`)
    5. Standardization (`standardize_mode`)
    6. Stack samples → batch tensor

    ???+ warning
        Pass `image_format` from `input_interface()` and `resize` / `standardize_mode`
        from `model.arch_resolution` / `model.arch_standardize` to ensure correct input
        shapes and normalization.

    ??? example "Example: How to integrate metadata in AUCMEDI?"
        ```python
        from aucmedi import *
        import numpy as np

        my_metadata = np.random.rand(len(samples), 10)

        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121",
                                  meta_variables=10)

        my_dg = DataGenerator(samples, "images_dir/",
                              labels=None, metadata=my_metadata,
                              resize=my_model.arch_resolution,                  # (224,224)
                              standardize_mode=my_model.arch_standardize)  # "torch"
        ```
    """

    # -----------------------------------------------------#
    #                    Initialization                   #
    # -----------------------------------------------------#
    def __init__(
        self,
        samples,
        path_imagedir,
        labels=None,
        metadata=None,
        image_format=None,
        subfunctions=[],
        batch_size=32,
        resize=(224, 224),
        standardize_mode="z-score",
        data_aug=None,
        shuffle=False,
        grayscale=False,
        two_dim=True,
        sample_weights=None,
        threads=1,
        prepare_images=False,
        loader=image_loader,
        seed=None,
        **kwargs,
    ):
        """Initialization function of the DataGenerator which acts as a configuration hub.

        If using for prediction, the 'labels' parameter has to be `None`.

        For more information on Subfunctions, read here: [aucmedi.data_processing.subfunctions][].

        Data augmentation is applied even for prediction if a DataAugmentation object is provided!

        ???+ warning
            Augmentation should only be applied to a **training** DataGenerator!

            For test-time augmentation, [aucmedi.ensemble.augmenting][] should be used.

        Applying `None` to `resize` will result into no image resizing. Default (224, 224)

        ???+ info "IO_loader Functions"
            | Interface                                                        | Description                                  |
            | ---------------------------------------------------------------- | -------------------------------------------- |
            | [image_loader()][aucmedi.data_processing.io_loader.image_loader] | Image Loader for image loading via Pillow. |
            | [sitk_loader()][aucmedi.data_processing.io_loader.sitk_loader]   | SimpleITK Loader for loading NIfTI (nii) or Metafile (mha) formats.    |
            | [numpy_loader()][aucmedi.data_processing.io_loader.numpy_loader] | NumPy Loader for image loading of .npy files.    |
            | [cache_loader()][aucmedi.data_processing.io_loader.cache_loader] | Cache Loader for passing already loaded images. |

            More information on IO_loader functions can be found here: [aucmedi.data_processing.io_loader][]. <br>
            Parameters defined in `**kwargs` are passed down to IO_loader functions.

        Args:
            samples (list of str):              List of sample/index encoded as Strings. Provided by
                                                [input_interface][aucmedi.data_processing.io_data.input_interface].
            path_imagedir (str):                Path to the directory containing the images.
            labels (numpy.ndarray):             Classification list with One-Hot Encoding. Provided by
                                                [input_interface][aucmedi.data_processing.io_data.input_interface].
            metadata (numpy.ndarray):           NumPy Array with additional metadata. Have to be shape (n_samples, meta_variables).
            image_format (str):                 Image format to add at the end of the sample index for image loading.
                                                Provided by [input_interface][aucmedi.data_processing.io_data.input_interface].
            subfunctions (List of Subfunctions):List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
            batch_size (int):                   Number of samples inside a single batch.
            resize (tuple of int):              Resizing shape consisting of a X and Y size. (optional Z size for Volumes)
            standardize_mode (str):             Standardization modus in which image intensity values are scaled.
                                                Calls the [Standardize][aucmedi.data_processing.subfunctions.standardize] Subfunction.
            data_aug (Augmentation Interface):  Data Augmentation class instance which performs diverse augmentation techniques.
                                                If `None` is provided, no augmentation will be performed.
            shuffle (bool):                     Boolean, whether dataset should be shuffled.
            grayscale (bool):                   Boolean, whether images are grayscale or RGB.
            sample_weights (list of float):     List of weights for samples. Can be computed via
                                                [compute_sample_weights()][aucmedi.utils.class_weights.compute_sample_weights].
            workers (int):                      Number of workers. If n_workers > 1 = use multi-threading for image preprocessing.
            prepare_images (bool):              Boolean, whether all images should be prepared and backup to disk before training.
                                                Recommended for large images or volumes to reduce CPU computing time.
            loader (io_loader function):        Function for loading samples/images from disk.
            seed (int):                         Seed to ensure reproducibility for random function.
            **kwargs (dict):                    Additional parameters for the sample loader.

        Attributes:
            has_labels (bool):              True if `labels` was provided (training / evaluation mode).
            has_metadata (bool):            True if `metadata` was provided.
            has_sample_weights (bool):      True if `sample_weights` was provided.
            samples (list of str):          The sample list as passed in.
            labels (numpy.ndarray or None): The label array as passed in.
            metadata (numpy.ndarray or None): The metadata array as passed in.
        """
        # Cache class variables
        self.samples = samples
        self.labels = labels
        self.has_labels = labels is not None
        self.metadata = metadata
        self.has_metadata = metadata is not None
        self.sample_weights = sample_weights
        self.has_sample_weights = sample_weights is not None
        self.prepare_images = prepare_images
        self.sample_loader = loader
        self.kwargs = kwargs
        self.path_imagedir = path_imagedir
        self.image_format = image_format
        self.grayscale = grayscale
        self.two_dim = two_dim
        self.subfunctions = subfunctions
        self.data_aug = data_aug
        self.standardize_mode = standardize_mode
        self.resize = resize
        self.seed = seed
        self.n = len(samples)
        self.index_array = None
        self.batch_size = batch_size if batch_size is not None else 1
        self.shuffle = shuffle
        self.threads = threads if threads is not None else 0

        self.max_iterations = (self.n + self.batch_size - 1) // self.batch_size
        self.iterations = self.max_iterations

        # Initialize Standardization Subfunction
        if standardize_mode is not None:
            self.sf_standardize = Standardize(mode=standardize_mode)
        else:
            self.sf_standardize = None
        # Validate resize shape against dimensionality and initialize Resizing Subfunction
        if resize is not None:
            try:
                rlen = len(resize)
            except TypeError:
                raise ValueError("`resize` must be a sequence with 2 or 3 elements")
            expected_len = 2 if self.two_dim else 3
            if rlen != expected_len:
                raise ValueError(
                    f"Parameter `resize` length {rlen} does not match expected "
                    f"dimension {expected_len} for two_dim={self.two_dim}: {resize}"
                )
            self.sf_resize = Resize(shape=resize)
        else:
            self.sf_resize = None
        # Sanity check for full sample list
        if samples is not None and len(samples) == 0:
            raise ValueError("Provided sample list is empty!", len(samples))
        # Sanity check for label correctness
        if labels is not None and len(samples) != len(labels):
            raise ValueError(
                "Samples and labels do not have same size!", len(samples), len(labels)
            )
        # Sanity check for metadata correctness
        if metadata is not None and len(samples) != len(metadata):
            raise ValueError(
                "Samples and metadata do not have same size!",
                len(samples),
                len(metadata),
            )
        # Sanity check for sample weights correctness
        if sample_weights is not None and len(samples) != len(sample_weights):
            raise ValueError(
                "Samples and sample weights do not have same size!",
                len(samples),
                len(sample_weights),
            )
        # Verify that labels, metadata and sample weights are NumPy arrays
        if labels is not None and not isinstance(labels, np.ndarray):
            self.labels = np.asarray(self.labels)
        if metadata is not None and not isinstance(metadata, np.ndarray):
            self.metadata = np.asarray(self.metadata)
        if sample_weights is not None and not isinstance(sample_weights, np.ndarray):
            self.sample_weights = np.asarray(self.sample_weights)

        # If prepare_image modus activated
        # -> Preprocess images beforehand and store them to disk for fast usage later
        if self.prepare_images:
            self.prepare_dir_object = tempfile.TemporaryDirectory(
                prefix="aucmedi.tmp.", suffix=".data"
            )
            self.prepare_dir = self.prepare_dir_object.name

            # Preprocess image for each index - Sequential
            if self.threads == 0 or self.threads == 1:
                for i in range(0, len(samples)):
                    self.preprocess_image(
                        index=i,
                        prepared_image=False,
                        run_resize=True,
                        run_aug=False,
                        run_standardize=False,
                        dump_pickle=True,
                    )
            # Preprocess image for each index - Multi-threading
            else:
                with ThreadPool(self.threads) as pool:
                    index_array = list(range(0, len(samples)))
                    mp_params = zip(
                        index_array,
                        repeat(False),
                        repeat(True),
                        repeat(False),
                        repeat(False),
                        repeat(True),
                    )
                    pool.starmap(self.preprocess_image, mp_params)
            print("A directory for image preparation was created:", self.prepare_dir)

    # -----------------------------------------------------#
    #              Batch Generation Function              #
    # -----------------------------------------------------#
    def _get_batches_of_transformed_samples(self, index_array):
        """Internal function for batch generation given a list of random selected samples."""
        # Initialize Batch stack
        batch_stack = ([],)
        if self.labels is not None:
            batch_stack += ([],)
        if self.sample_weights is not None:
            batch_stack += ([],)

        # Process image for each index - Sequential
        if self.threads == 0 or self.threads == 1:
            for i in index_array:
                batch_img = self.preprocess_image(
                    index=i, prepared_image=self.prepare_images
                )
                batch_stack[0].append(batch_img)
        # Process image for each index - Multi-threading
        else:
            with ThreadPool(self.threads) as pool:
                mp_params = zip(index_array, repeat(self.prepare_images))
                batches_img = pool.starmap(self.preprocess_image, mp_params)
            batch_stack[0].extend(batches_img)

        # Add classification to batch if available
        if self.labels is not None:
            batch_stack[1].extend(self.labels[index_array])
        # Add sample weight to batch if available
        if self.sample_weights is not None:
            batch_stack[2].extend(self.sample_weights[index_array])

        # Stack images and optional metadata together into a batch
        input_stack = np.stack(batch_stack[0], axis=0)
        if self.metadata is not None:
            input_stack = (input_stack, self.metadata[index_array])
        batch = (input_stack,)
        # Stack classifications together into a batch if available
        if self.labels is not None:
            batch += (np.stack(batch_stack[1], axis=0),)
        # Stack sample weights together into a batch if available
        if self.sample_weights is not None:
            batch += (np.stack(batch_stack[2], axis=0),)
        # Return generated Batch
        if len(batch) == 1:
            # Conventionally, PyTorch DataLoader expects the Dataset to return a single item (the batch), so we unwrap it from the tuple.
            return batch[0]
        return batch

    # -----------------------------------------------------#
    #                 Image Preprocessing                 #
    # -----------------------------------------------------#
    def preprocess_image(
        self,
        index,
        prepared_image=False,
        run_resize=True,
        run_aug=True,
        run_standardize=True,
        dump_pickle=False,
    ):
        """Internal preprocessing function for applying Subfunctions, augmentation, resizing and standardization
        on an image given its index.

        Activating the prepared_image option also allows loading a beforehand preprocessed image from disk.

        Deactivating the run_aug & run_standardize option to output image without augmentation and standardization.

        Activating dump_pickle will store the preprocessed image as pickle on disk instead of returning.
        """
        # Load prepared image from disk
        if prepared_image:
            # Load from disk
            path_img = os.path.join(self.prepare_dir, "img_" + str(index))
            with open(path_img + ".pickle", "rb") as pickle_loader:
                img = pickle.load(pickle_loader)
            # Apply image augmentation on image if activated
            if self.data_aug is not None and run_aug:
                img = self.data_aug.apply(img)
            # Apply standardization on image if activated
            if self.sf_standardize is not None and run_standardize:
                img = self.sf_standardize.transform(img)
        # Preprocess image during runtime
        else:
            # Load image from disk
            img = self.sample_loader(
                self.samples[index],
                self.path_imagedir,
                image_format=self.image_format,
                grayscale=self.grayscale,
                two_dim=self.two_dim,
                **self.kwargs,
            )
            # Apply subfunctions on image
            for sf in self.subfunctions:
                img = sf.transform(img)
            # Apply resizing on image if activated
            if self.sf_resize is not None and run_resize:
                img = self.sf_resize.transform(img)
            # Apply image augmentation on image if activated
            if self.data_aug is not None and run_aug:
                img = self.data_aug.apply(img)
            # Apply standardization on image if activated
            if self.sf_standardize is not None and run_standardize:
                img = self.sf_standardize.transform(img)
        # Dump preprocessed image to disk (for later usage via prepared_image)
        if dump_pickle:
            path_img = os.path.join(self.prepare_dir, "img_" + str(index))
            with open(path_img + ".pickle", "wb") as pickle_writer:
                pickle.dump(img, pickle_writer)
        # Return preprocessed image in channel-first format (C,H,W) / (C,D,H,W)
        else:
            if img.ndim == 3:    # 2D: (H, W, C) -> (C, H, W)
                img = np.transpose(img, (2, 0, 1))
            elif img.ndim == 4:  # 3D: (D, H, W, C) -> (C, D, H, W)
                img = np.transpose(img, (3, 0, 1, 2))
            return img

    # -----------------------------------------------------#
    #              Sample Generation Function             #
    # -----------------------------------------------------#
    def __getitem__(self, raw_idx):
        """Internal function for calling the batch generation process."""
        # Obtain the index based on the passed index offset to allow repetition
        idx = raw_idx % self.max_iterations
        # Build index array for the start
        if self.index_array is None:
            self.__set_index_array__()
        # Select samples for next batch
        index_array = self.index_array[
            self.batch_size * idx : self.batch_size * (idx + 1)
        ]
        # Generate batch
        return self._get_batches_of_transformed_samples(index_array)

    # -----------------------------------------------------#
    #                 Generator Functions                 #
    # -----------------------------------------------------#
    def __len__(self):
        """Internal function for identifying the generator length."""
        return self.iterations

    def set_length(self, iterations):
        """Configuration function for fixing the number of iterations."""
        self.iterations = iterations

    def reset_length(self):
        """Configuration function for reseting the number of iterations."""
        self.iterations = self.max_iterations

    def __set_index_array__(self):
        """Internal function for initializing and shuffling the index array."""
        # Generate index array
        self.index_array = np.arange(self.n)
        # Shuffle if needed
        if self.shuffle:
            # Update seed for repeated permutation of the index_array
            if self.seed is not None:
                np.random.seed(self.seed + self.seed_walk)
                self.seed_walk += 1
            # Permutate index array
            self.index_array = np.random.permutation(self.n)

    """ Internal function at the end of an epoch. """

    def on_epoch_end(self):
        self.__set_index_array__()
