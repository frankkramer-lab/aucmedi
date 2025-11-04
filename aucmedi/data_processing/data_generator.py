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
import torch
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
class DataGenerator(Dataset):
    """Infinite Data Generator which automatically creates batches from a list of samples.

    The created batches are model ready. This generator can be supplied directly
    to a [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork] train() & predict()
    function (also compatible to tensorflow.keras.model fit() & predict() function).

    The DataGenerator is the second of the three pillars of AUCMEDI.

    ??? info "Pillars of AUCMEDI"
        - [aucmedi.data_processing.io_data.input_interface][]
        - [aucmedi.data_processing.data_generator.DataGenerator][]
        - [aucmedi.neural_network.model.NeuralNetwork][]

    The DataGenerator can be used for training, validation as well as for prediction.

    ???+ example
        ```python
        # Import
        from aucmedi import *

        # Initialize model
        model = NeuralNetwork(
            n_labels=8,
            channels=3,
            architecture="2D.ResNet50"
        )

        # Do some training
        datagen_train = DataGenerator(
            samples=samples[:100],
            path_imagedir="images_dir/",
            image_format=image_format,
            labels=class_ohe[:100],
            resize=model.meta_input,
            standardize_mode=model.meta_standardize
        )

        model.train(datagen_train, epochs=50)

        # Do some predictions
        datagen_test = DataGenerator(
            samples=samples[100:150],
            path_imagedir="images_dir/",
            image_format=image_format,
            labels=None,
            resize=model.meta_input,
            standardize_mode=model.meta_standardize
        )

        preds = model.predict(datagen_test)
        ```

    It supports real-time batch generation as well as beforehand preprocessing of images,
    which are then temporarily stored on disk (requires enough disk space!).

    The resulting batches are created based the following pipeline:

    1. Image Loading
    2. Application of Subfunctions
    3. Resize image
    4. Application of Data Augmentation
    5. Standardize image
    6. Stacking processed images to a batch

    ???+ warning
        When instantiating a `DataGenerator`, it is highly recommended, to pass the `image_format` parameter provided
        by the `input_interface()` and the `resize` & `standardize_mode` parameters provided by the
        `NeuralNetwork` class attributes `meta_input` & `meta_standardize`.

        It assures, that the samples contain the expected file extension, input shape and standardization.

    ???+ abstract "Build on top of the library"
        Tensorflow.Keras Iterator: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/Iterator

    ??? example "Example: How to integrate metadata in AUCMEDI?"
        ```python
        from aucmedi import *
        import numpy as np

        my_metadata = np.random.rand(len(samples), 10)

        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121",
                                  meta_variables=10)

        my_dg = DataGenerator(samples, "images_dir/",
                              labels=None, metadata=my_metadata,
                              resize=my_model.meta_input,                  # (224,224)
                              standardize_mode=my_model.meta_standardize)  # "torch"
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
        resize=(224, 224),
        standardize_mode="z-score",
        data_aug=None,
        grayscale=False,
        sample_weights=None,
        prepare_images=False,
        loader=image_loader,
        seed=None,
        **kwargs
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
            resize (tuple of int):              Resizing shape consisting of a X and Y size. (optional Z size for Volumes)
            standardize_mode (str):             Standardization modus in which image intensity values are scaled.
                                                Calls the [Standardize][aucmedi.data_processing.subfunctions.standardize] Subfunction.
            data_aug (Augmentation Interface):  Data Augmentation class instance which performs diverse augmentation techniques.
                                                If `None` is provided, no augmentation will be performed.
            grayscale (bool):                   Boolean, whether images are grayscale or RGB.
            sample_weights (list of float):     List of weights for samples. Can be computed via
                                                [compute_sample_weights()][aucmedi.utils.class_weights.compute_sample_weights].
            workers (int):                      Number of workers. If n_workers > 1 = use multi-threading for image preprocessing.
            prepare_images (bool):              Boolean, whether all images should be prepared and backup to disk before training.
                                                Recommended for large images or volumes to reduce CPU computing time.
            loader (io_loader function):        Function for loading samples/images from disk.
            seed (int):                         Seed to ensure reproducibility for random function.
            **kwargs (dict):                    Additional parameters for the sample loader.
        """
        # Cache class variables
        self.samples = samples
        self.labels = labels
        self.metadata = metadata
        self.sample_weights = sample_weights
        self.prepare_images = prepare_images
        self.sample_loader = loader
        self.kwargs = kwargs
        self.path_imagedir = path_imagedir
        self.image_format = image_format
        self.grayscale = grayscale
        self.subfunctions = subfunctions
        self.data_aug = data_aug
        self.standardize_mode = standardize_mode
        self.resize = resize
        self.seed = seed

        # Initialize Standardization Subfunction
        if standardize_mode is not None:
            self.sf_standardize = Standardize(mode=standardize_mode)
        else:
            self.sf_standardize = None
        # Initialize Resizing Subfunction
        if resize is not None:
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
            for i in range(0, len(samples)):
                self.preprocess_image(
                    index=i,
                    prepared_image=False,
                    run_resize=True,
                    run_aug=False,
                    run_standardize=False,
                    dump_pickle=True,
                )
            print("A directory for image preparation was created:", self.prepare_dir)

    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------#
    #              Sample Generation Function             #
    # -----------------------------------------------------#

    def __getitem__(self, index: int):
        """Return a single preprocessed sample (and optional label/metadata/sample_weight)."""
        # Preprocess / load the image
        img = self.preprocess_image(index=index, prepared_image=self.prepare_images)

        # Convert numpy array to torch tensor
        img = torch.from_numpy(img).float()

        # Build input (include metadata if available)
        if self.metadata is not None:
            metadata = torch.from_numpy(self.metadata[index]).float()
            input_item = (img, metadata)
        else:
            input_item = img

        # Assemble return tuple similar to batch output structure
        result = (input_item,)
        if self.labels is not None:
            label = torch.from_numpy(self.labels[index]).float()
            result += (label,)
        if self.sample_weights is not None:
            weight = torch.tensor(self.sample_weights[index], dtype=torch.float)
            result += (weight,)

        return result

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
                **self.kwargs
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
        # Return preprocessed image
        else:
            return img
