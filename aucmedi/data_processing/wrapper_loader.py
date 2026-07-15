# ==============================================================================#
#  Author:       Fabian Wehr                                               #
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import warnings
from functools import partial

from aucmedi.data_processing.batch_generator import BatchGenerator
from aucmedi.data_processing.io_loader.image_loader import image_loader


def create_batch_loader(
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
    num_workers=0,
    **kwargs
):
    """Creates a BatchGenerator with specified parameters and wraps it in a DataLoader.
    Args:
        samples (list of str):              List of sample/index encoded as Strings.
        path_imagedir (str):                Path to the directory containing the images.
        labels (numpy.ndarray):             Classification list with One-Hot Encoding.
        metadata (numpy.ndarray):           NumPy Array with additional metadata.
        image_format (str):                 Image format to add at the end of the sample index for image loading.
        subfunctions (List of Subfunctions):List of Subfunctions class instances.
        batch_size (int):                   Number of samples inside a single batch.
        resize (tuple of int):              Resizing shape consisting of a X and Y size.
        standardize_mode (str):             Standardization modus in which image intensity values are scaled.
        data_aug (Augmentation Interface):  Data Augmentation class instance.
        shuffle (bool):                     Boolean, whether dataset should be shuffled.
        grayscale (bool):                   Boolean, whether images are grayscale or RGB.
        two_dim (bool):                     Boolean, whether images are two-dimensional.
        sample_weights (list of float):     List of weights for samples.
        threads (int):                      Number of workers for image preprocessing.
        prepare_images (bool):              Boolean, whether all images should be prepared and backup to disk
                                            before training.
        loader (io_loader function):        Function for loading samples/images from disk.
        seed (int):                         Seed to ensure reproducibility for random function.
        num_workers (int):                  Number of workers for DataLoader.
        **kwargs (dict):                    Additional parameters for the sample loader.
    Returns:
        WrapperLoader: A DataLoader wrapping the BatchGenerator.
    """
    # Initialize BatchGenerator
    batch_gen = BatchGenerator(
        samples=samples,
        path_imagedir=path_imagedir,
        labels=labels,
        metadata=metadata,
        image_format=image_format,
        subfunctions=subfunctions,
        batch_size=batch_size,
        resize=resize,
        standardize_mode=standardize_mode,
        data_aug=data_aug,
        shuffle=shuffle,
        grayscale=grayscale,
        two_dim=two_dim,
        sample_weights=sample_weights,
        threads=threads,
        prepare_images=prepare_images,
        loader=loader,
        seed=seed,
        **kwargs
    )
    # Initialize WrapperLoader
    wrapper_loader = WrapperLoader(batch_gen, num_workers=num_workers, **kwargs)
    return wrapper_loader


def _shard_and_create_batch_loader(
    rank,
    world_size,
    samples,
    path_imagedir,
    labels=None,
    metadata=None,
    sample_weights=None,
    **kwargs,
):
    """Builds this rank's shard (samples[rank::world_size]) as a WrapperLoader.

    Internal helper behind [create_distributed_loader()][aucmedi.data_processing.wrapper_loader.create_distributed_loader].
    Must stay a top-level function (not a closure) so torch.multiprocessing.spawn
    can pickle the `functools.partial` built around it.
    """
    return create_batch_loader(
        samples=samples[rank::world_size],
        path_imagedir=path_imagedir,
        labels=labels[rank::world_size] if labels is not None else None,
        metadata=metadata[rank::world_size] if metadata is not None else None,
        sample_weights=sample_weights[rank::world_size]
        if sample_weights is not None
        else None,
        **kwargs,
    )


def create_distributed_loader(
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
    num_workers=0,
    **kwargs,
):
    """Builds a `(rank, world_size) -> WrapperLoader` factory for [NeuralNetwork.train_distributed()][aucmedi.neural_network.model.NeuralNetwork.train_distributed].

    Takes the exact same arguments as [create_batch_loader()][aucmedi.data_processing.wrapper_loader.create_batch_loader],
    but instead of building one loader immediately, it returns a callable that
    shards `samples`/`labels`/`metadata`/`sample_weights` by `[rank::world_size]`
    and calls `create_batch_loader()` for that shard only. `train_distributed()`
    calls this factory once inside each spawned worker process, so every rank
    builds its own `BatchGenerator`/`ThreadPool`/`WrapperLoader` instead of
    sharing one (unpicklable) instance across processes.

    ???+ example
        ```python
        from aucmedi.data_processing.wrapper_loader import create_distributed_loader

        train_loader_fn = create_distributed_loader(
            samples, "images_dir/", labels=class_ohe,
            resize=model.arch_resolution, standardize_mode=model.arch_standardize,
        )
        model.train_distributed(generator_fn=train_loader_fn, epochs=50)
        ```

    Args:
        samples (list of str):              List of sample/index encoded as Strings.
        path_imagedir (str):                Path to the directory containing the images.
        labels (numpy.ndarray):             Classification list with One-Hot Encoding.
        metadata (numpy.ndarray):           NumPy Array with additional metadata.
        image_format (str):                 Image format to add at the end of the sample index for image loading.
        subfunctions (List of Subfunctions):List of Subfunctions class instances.
        batch_size (int):                   Number of samples inside a single batch.
        resize (tuple of int):              Resizing shape consisting of a X and Y size.
        standardize_mode (str):             Standardization modus in which image intensity values are scaled.
        data_aug (Augmentation Interface):  Data Augmentation class instance.
        shuffle (bool):                     Boolean, whether dataset should be shuffled.
        grayscale (bool):                   Boolean, whether images are grayscale or RGB.
        two_dim (bool):                     Boolean, whether images are two-dimensional.
        sample_weights (list of float):     List of weights for samples.
        threads (int):                      Number of workers for image preprocessing.
        prepare_images (bool):              Boolean, whether all images should be prepared and backup to disk
                                            before training.
        loader (io_loader function):        Function for loading samples/images from disk.
        seed (int):                         Seed to ensure reproducibility for random function.
        num_workers (int):                  Number of workers for DataLoader.
        **kwargs (dict):                    Additional parameters for the sample loader.
    Returns:
        Callable[[int, int], WrapperLoader]: Factory taking `(rank, world_size)`.
    """
    return partial(
        _shard_and_create_batch_loader,
        samples=samples,
        path_imagedir=path_imagedir,
        labels=labels,
        metadata=metadata,
        image_format=image_format,
        subfunctions=subfunctions,
        batch_size=batch_size,
        resize=resize,
        standardize_mode=standardize_mode,
        data_aug=data_aug,
        shuffle=shuffle,
        grayscale=grayscale,
        two_dim=two_dim,
        sample_weights=sample_weights,
        threads=threads,
        prepare_images=prepare_images,
        loader=loader,
        seed=seed,
        num_workers=num_workers,
        **kwargs,
    )


# ------------------------------------------------------------------#
# Helper: Convert numpy structures to torch.Tensor recursively      #
# ------------------------------------------------------------------#
def _to_torch(obj):
    """Recursively convert numpy arrays / nested lists/tuples to torch.Tensor.
    Leaves torch.Tensors unchanged.
    """
    # Torch tensor -> return as-is
    if isinstance(obj, torch.Tensor):
        return obj
    # NumPy array -> convert to float tensor (float32)
    if isinstance(obj, np.ndarray):
        # Convert to contiguous array then to torch
        tensor = torch.from_numpy(np.ascontiguousarray(obj)).float()
        return tensor
    # List/tuple -> recursively convert and preserve type
    if isinstance(obj, list):
        return [_to_torch(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_torch(x) for x in obj)
    # Other types -> pass through
    return obj


# -----------------------------------------------------#
#                 Data Loader                #
# -----------------------------------------------------#
class WrapperLoader(DataLoader):
    """Custom DataLoader class which extends the PyTorch DataLoader class.

    This class is a thin wrapper around the PyTorch DataLoader class, which
    utilizes our custom [Dataset][aucmedi.data_processing.batch_generator.BatchGenerator]
    class for loading images/volumes and applying preprocessing steps.

    Note:
        The wrapped BatchGenerator already returns full batches. To avoid double-
        batching by torch.utils.data.DataLoader we force `batch_size=None`.
    Note:
        The recommended way to initialize this class is to call the [create_batch_loader()][aucmedi.data_processing.wrapper_loader.create_batch_loader]
        function which initializes the BatchGenerator and wraps it in a WrapperLoader.
    """

    def __init__(self, batch_generator: BatchGenerator, **kwargs):
        """Initialization function for creating a DataLoader which can be passed to
            [NeuralNetwork.train()][aucmedi.neural_network.model.NeuralNetwork.train]
            or [NeuralNetwork.predict()][aucmedi.neural_network.model.NeuralNetwork.predict].

        Args:
            batch_generator (BatchGenerator): A [BatchGenerator][aucmedi.data_processing.batch_generator.BatchGenerator]
                object which inherits from PyTorch Dataset class and provides
                functionality to load images/volumes and apply preprocessing steps.
        """
        # Initialize DataLoader values from batch_generator for compatibility and utility access
        self.dataset = batch_generator
        self.data_aug = batch_generator.data_aug
        self.batch_size = batch_generator.batch_size
        self.shuffle = batch_generator.shuffle
        self.has_labels = batch_generator.has_labels
        self.has_metadata = batch_generator.has_metadata
        self.has_sample_weights = batch_generator.has_sample_weights
        self.kwargs = kwargs

        # Extract relevant kwargs for DataLoader
        self.num_workers = kwargs.pop("num_workers", 0)
        if self.num_workers <= 1:
            self.num_workers = 0  # Force single-threaded operation
        else:
            self.prefetch_factor = kwargs.pop("prefetch_factor", 2)
        self.pin_memory = kwargs.pop("pin_memory", False)
        self.timeout = kwargs.pop("timeout", 0)
        self.worker_init_fn = kwargs.pop("worker_init_fn", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", None)
        if self.num_workers > 0 and self.multiprocessing_context is None:
            # Forking a process that already initialized a CUDA context hangs
            # (undefined CUDA driver state in the forked worker). Since NeuralNetwork
            # moves the model to the GPU before building this DataLoader, "fork"
            # (the default multiprocessing start method on Linux) is unsafe here.
            self.multiprocessing_context = "spawn"
        self.generator = kwargs.pop("generator", None)
        self.persistent_workers = kwargs.pop("persistent_workers", False)
        self.pin_memory_device = kwargs.pop("pin_memory_device", "cuda")
        # Extract collate_fn if provided
        collate_fn = kwargs.pop("collate_fn", None)
        if collate_fn is not None:
            # Alert user that their collate_fn will be ignored
            warnings.warn(
                "Custom collate_fn is ignored because BatchGenerator already returns full batches"
            )

        # Initialize PyTorch DataLoader with batch_size=None to avoid re-batching.
        super().__init__(
            dataset=batch_generator,  # dataset
            batch_size=None,  # dataset already returns a batch
            shuffle=False,  # underlying generator handles shuffling
            sampler=None,
            batch_sampler=None,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=self.multiprocessing_context,
            generator=self.generator,
            persistent_workers=self.persistent_workers,
            pin_memory_device=self.pin_memory_device,
        )

        # Keep reference to underlying generator/dataset for utility access
        self.batch_generator = batch_generator

    # -----------------------------------------------------#
    # Iteration helpers (override torch DataLoader behavior)
    # -----------------------------------------------------#
    def __iter__(self):
        """Iterate over underlying BatchGenerator and yield full batches.

        Convert numpy-based batches to torch tensors for PyTorch compatibility.
        """
        for i in range(len(self.batch_generator)):
            batch = self.batch_generator[i]
            # Convert numpy arrays to torch tensors
            batch = _to_torch(batch)
            yield batch

    def __len__(self):
        """Return number of iterations (batches) per epoch."""
        return len(self.batch_generator)

    # ------------------------------------------------------------------#
    # Proxy utilities to control the underlying BatchGenerator functions  #
    # ------------------------------------------------------------------#
    def set_length(self, iterations):
        """Proxy to BatchGenerator.set_length"""
        if hasattr(self.batch_generator, "set_length"):
            return self.batch_generator.set_length(iterations)
        raise AttributeError("Underlying dataset has no method set_length")

    def reset_length(self):
        """Proxy to BatchGenerator.reset_length"""
        if hasattr(self.batch_generator, "reset_length"):
            return self.batch_generator.reset_length()
        raise AttributeError("Underlying dataset has no method reset_length")

    def on_epoch_end(self):
        """Proxy to BatchGenerator.on_epoch_end"""
        if hasattr(self.batch_generator, "on_epoch_end"):
            return self.batch_generator.on_epoch_end()
        raise AttributeError("Underlying dataset has no method on_epoch_end")

    def preprocess_image(self, *args, **kwargs):
        """Proxy to BatchGenerator.preprocess_image"""
        if hasattr(self.batch_generator, "preprocess_image"):
            return self.batch_generator.preprocess_image(*args, **kwargs)
        raise AttributeError("Underlying dataset has no method preprocess_image")

    # Optionally expose samples/metadata access
    @property
    def samples(self):
        return getattr(self.batch_generator, "samples", None)

    @property
    def labels(self):
        return getattr(self.batch_generator, "labels", None)

    @property
    def metadata(self):
        return getattr(self.batch_generator, "metadata", None)
