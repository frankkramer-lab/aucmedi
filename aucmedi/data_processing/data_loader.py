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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import repeat
import tempfile
import pickle
import os
import torch

# Internal libraries
from aucmedi.data_processing.io_loader import image_loader
from aucmedi.data_processing.subfunctions import Standardize, Resize
from aucmedi.data_processing import data_generator


# Picklable passthrough collate function used when the Dataset already returns full batches.
def _passthrough_collate(batch):
    """Return the single item (already a batch) provided by the Dataset.

    This function must be defined at module level so it is picklable by multiprocessing.
    """
    return batch[0]


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
        return torch.from_numpy(np.ascontiguousarray(obj)).float()
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
class data_loader(DataLoader):
    """Custom DataLoader class which extends the PyTorch DataLoader class.

    This class is a thin wrapper around the PyTorch DataLoader class, which
    utilizes our custom [Dataset][aucmedi.data_processing.data_generator.DataGenerator]
    class for loading images/volumes and applying preprocessing steps.

    Note:
        The wrapped DataGenerator already returns full batches. To avoid double-
        batching by torch.utils.data.DataLoader we force `batch_size=None` and
        use a collate_fn that returns the single dataset item unchanged.
    """

    def __init__(self, data_generator: Dataset):
        """Initialization function for creating a DataLoader which can be passed to
            [NeuralNetwork.train()][aucmedi.neural_network.model.NeuralNetwork.train]
            or [NeuralNetwork.predict()][aucmedi.neural_network.model.NeuralNetwork.predict].

        Args:
            data_generator (Dataset): A [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator]
                object which inherits from PyTorch Dataset class and provides
                functionality to load images/volumes and apply preprocessing steps.
        """
        # Initialize DataLoader values from data_generator
        dataset = data_generator
        batch_size = data_generator.batch_size
        shuffle = data_generator.shuffle

        # Map remaining DataLoader parameters from provided data_generator
        sampler = getattr(data_generator, "sampler", None)
        batch_sampler = getattr(data_generator, "batch_sampler", None)
        num_workers = getattr(data_generator, "num_workers", 0)
        pin_memory = getattr(data_generator, "pin_memory", False)
        drop_last = getattr(data_generator, "drop_last", False)
        timeout = getattr(data_generator, "timeout", 0)
        worker_init_fn = getattr(data_generator, "worker_init_fn", None)
        multiprocessing_context = getattr(
            data_generator, "multiprocessing_context", None
        )
        generator = getattr(data_generator, "generator", None)
        prefetch_factor = getattr(data_generator, "prefetch_factor", None)
        persistent_workers = getattr(data_generator, "persistent_workers", False)
        pin_memory_device = getattr(data_generator, "pin_memory_device", "")

        # If no custom collate_fn provided by user, use the picklable module-level passthrough
        collate_fn = _passthrough_collate

        # Initialize PyTorch DataLoader with batch_size=None to avoid re-batching.
        super().__init__(
            dataset=data_generator,  # dataset
            batch_size=None,  # dataset already returns a batch
            shuffle=False,  # underlying generator handles shuffling
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

        # Keep reference to underlying generator/dataset for utility access
        self.data_generator = data_generator

    # -----------------------------------------------------#
    # Iteration helpers (override torch DataLoader behavior)
    # -----------------------------------------------------#
    def __iter__(self):
        """Iterate over underlying DataGenerator and yield full batches.

        Convert returned numpy-based batches to torch.Tensor objects so that
        Keras' torch_data_loader_adapter can call .cpu() on elements.
        """
        for i in range(len(self.data_generator)):
            batch = self.data_generator[i]
            yield _to_torch(batch)

    def __len__(self):
        """Return number of iterations (batches) per epoch."""
        return len(self.data_generator)

    # ------------------------------------------------------------------#
    # Proxy utilities to control the underlying DataGenerator functions  #
    # ------------------------------------------------------------------#
    def set_length(self, iterations):
        """Proxy to DataGenerator.set_length"""
        if hasattr(self.data_generator, "set_length"):
            return self.data_generator.set_length(iterations)
        raise AttributeError("Underlying dataset has no method set_length")

    def reset_length(self):
        """Proxy to DataGenerator.reset_length"""
        if hasattr(self.data_generator, "reset_length"):
            return self.data_generator.reset_length()
        raise AttributeError("Underlying dataset has no method reset_length")

    def on_epoch_end(self):
        """Proxy to DataGenerator.on_epoch_end"""
        if hasattr(self.data_generator, "on_epoch_end"):
            return self.data_generator.on_epoch_end()
        raise AttributeError("Underlying dataset has no method on_epoch_end")

    def preprocess_image(self, *args, **kwargs):
        """Proxy to DataGenerator.preprocess_image"""
        if hasattr(self.data_generator, "preprocess_image"):
            return self.data_generator.preprocess_image(*args, **kwargs)
        raise AttributeError("Underlying dataset has no method preprocess_image")

    # Optionally expose samples/metadata access
    @property
    def samples(self):
        return getattr(self.data_generator, "samples", None)

    @property
    def labels(self):
        return getattr(self.data_generator, "labels", None)

    @property
    def metadata(self):
        return getattr(self.data_generator, "metadata", None)
