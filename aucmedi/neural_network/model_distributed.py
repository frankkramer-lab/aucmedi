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
from time import time
import numpy as np
import os
import tempfile

import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Internal libraries/scripts
from aucmedi.neural_network.architectures import (
    architecture_dict,
    supported_standardize_mode,
    Classifier,
)
from aucmedi.utils.callbacks import Callback, EarlyStoppingCallback

def ddp_setup(rank, world_size, device_id, backend="nccl"):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        device_id: CUDA device index this rank should use (may repeat across
                   ranks when simulating more ranks than physical GPUs)
        backend: torch.distributed backend. Use "gloo" instead of "nccl" when
                 multiple ranks share the same GPU (NCCL does not support that).
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Windows PyTorch builds commonly lack libuv support, which TCPStore
    # defaults to since PyTorch 2.x; disable it so init_process_group works
    # cross-platform (no effect on builds that do have libuv).
    os.environ.setdefault("USE_LIBUV", "0")
    torch.cuda.set_device(device_id)
    init_process_group(backend=backend, rank=rank, world_size=world_size)


# -----------------------------------------------------#
#            Neural Network (model) class             #
# -----------------------------------------------------#
# Class which represents the Neural Network
class NeuralNetwork:
    """ Neural Network class providing functionality for handling all model methods.

    This class is the third of the three pillars of AUCMEDI.

    ??? info "Pillars of AUCMEDI"
        - [aucmedi.data_processing.io_data.input_interface][]
        - [aucmedi.data_processing.batch_generator.BatchGenerator][]
        - [aucmedi.neural_network.model.NeuralNetwork][]

    With an initialized Neural Network model instance, it is possible to run training and predictions.

    ??? example "Example: How to use"
        ```python
        from aucmedi import *
        from aucmedi.data_processing.wrapper_loader import create_batch_loader

        # Initialize model
        model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.ResNet50")
        # Do some training
        train_loader = create_batch_loader(samples[:100], "images_dir/", labels=class_ohe[:100],
                                           resize=model.arch_resolution, standardize_mode=model.arch_standardize)
        model.train(train_loader, epochs=50)
        # Do some predictions
        test_loader = create_batch_loader(samples[100:150], "images_dir/", labels=None,
                                          resize=model.arch_resolution, standardize_mode=model.arch_standardize)
        preds = model.predict(test_loader)
        ```

    ??? example "Example: How to select an Architecture"
        ```python
        # 2D architecture
        my_model_a = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")
        # 3D architecture for multi-label classification (sigmoid activation)
        my_model_b = NeuralNetwork(n_labels=8, channels=3, architecture="3D.ResNet50",
                                    activation_output="sigmoid")
        # 2D architecture with custom input_resolution
        my_model_c = NeuralNetwork(n_labels=8, channels=3, architecture="2D.ConvNeXtBase",
                                    input_resolution=(512,512))
        ```

    ??? note "List of implemented Architectures"
        AUCMEDI provides a large library of state-of-the-art and ready-to-use architectures.

        - 2D Architectures: [aucmedi.neural_network.architectures.image][]
        - 3D Architectures: [aucmedi.neural_network.architectures.volume][]

    ??? note "Classification Types"
        | Type                       | Activation Function                                             |
        | -------------------------- | --------------------------------------------------------------- |
        | Binary classification      | `activation_output="softmax"`: Only a single class is correct.  |
        | Multi-class classification | `activation_output="softmax"`: Only a single class is correct.  |
        | Multi-label classification | `activation_output="sigmoid"`: Multiple classes can be correct. |

        Defined by the [Classifier][aucmedi.neural_network.architectures.classifier] of an
        [Architecture][aucmedi.neural_network.architectures].

    ??? example "Example: How to obtain required parameters for the BatchGenerator?"
        Be aware that the input_size and standardize_mode are just recommendations and
        can be changed by desire. <br>
        However, the recommended parameter are required for transfer learning.

        ```python title="Recommended way"
        from aucmedi.data_processing.wrapper_loader import create_batch_loader

        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")

        my_loader = create_batch_loader(samples, "images_dir/", labels=None,
                                        resize=my_model.arch_resolution,                  # (224,224)
                                        standardize_mode=my_model.arch_standardize)       # "torch"
        ```

        ```python title="Manual way"
        from aucmedi.neural_network.architectures import Classifier, \
                                                         architecture_dict, \
                                                         supported_standardize_mode
        from aucmedi.data_processing.wrapper_loader import create_batch_loader

        my_arch = architecture_dict["3D.DenseNet121"](n_labels=4,
                                                      channels=1,
                                                      input_resolution=(128,128,128))

        my_model = NeuralNetwork(n_labels=None, channels=None, architecture=my_arch)

        from aucmedi.neural_network.architectures import supported_standardize_mode
        sf_norm = supported_standardize_mode["3D.DenseNet121"]
        my_loader = create_batch_loader(samples, "images_dir/", labels=None,
                                        resize=(128,128,128),                        # (128,128,128)
                                        standardize_mode=sf_norm)                    # "torch"
        ```

    ??? example "Example: How to integrate metadata in AUCMEDI?"
        ```python
        from aucmedi import *
        from aucmedi.data_processing.wrapper_loader import create_batch_loader
        import numpy as np

        my_metadata = np.random.rand(len(samples), 10)

        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121",
                                  n_meta_variables=10)

        my_loader = create_batch_loader(samples, "images_dir/",
                                        labels=None, metadata=my_metadata,
                                        resize=my_model.arch_resolution,                  # (224,224)
                                        standardize_mode=my_model.arch_standardize)       # "torch"
        ```
    """

    def __init__(
        self,
        n_labels,
        channels,
        input_resolution=None,
        architecture=None,
        pretrained_weights=False,
        loss=None,
        metrics=None,
        activation_output="softmax",
        fcl_dropout=True,
        n_meta_variables=None,
        verbose=1,
    ):
        """Initialization function for creating a Neural Network (model) object.

        Args:
            n_labels (int):                         Number of classes/labels (important for the last layer).
            channels (int):                         Number of channels. Grayscale:1 or RGB:3.
            input_resolution (tuple):               Input resolution of the batch imaging data (excluding channel axis).
                                                    If None is provided, the default input_resolution for the architecture is selected
                                                    from the architecture dictionary.
            architecture (str or Architecture):     Key (str) or instance of a neural network model Architecture class instance.
                                                    If a string is provided, the corresponding architecture is selected from the architecture dictionary.
                                                    A string has to begin with either '3D.' or '2D' depending on the classification task.
                                                    By default, a 2D Vanilla Model is used as architecture.
            pretrained_weights (bool):              Option whether to utilize pretrained weights e.g. from ImageNet.
            loss (Metric Function):                 The loss function which is used for training.
                                                    Any loss function defined in PyTorch or aucmedi.neural_network.loss_functions can be used.
            metrics (list of Metric Functions):     List of one or multiple metric functions for evaluation.
                                                    Any metric function defined in PyTorch or custom functions can be used.
            activation_output (str):                Activation function which is used during prediction.
            fcl_dropout (bool):                     Option whether to utilize an additional Linear & Dropout layer in the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            n_meta_variables (int):                 Number of metadata variables, which should be included in the classification head.
                                                    If `None`is provided, no metadata integration block will be added to the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
            verbose (int):                          Option (0/1) how much information should be written to stdout.

        ???+ danger
            Class attributes can be modified also after initialization, at will.
            However, be aware of unexpected adverse effects (experimental)!

        Attributes:
            arch_resolution (tuple of int):         Meta variable: Input resolution of architecture which can be passed to a DataGenerator. For example: (224, 224).
            arch_standardize (str):                 Meta variable: Recommended standardize_mode of architecture which can be passed to a DataGenerator.
                                                    For example: "torch".
        """
        # Cache parameters
        self.n_labels = n_labels
        self.channels = channels
        self.loss = (
            loss
            if loss is not None and isinstance(loss, nn.Module)
            else torch.nn.CrossEntropyLoss()
        )
        self.metrics = metrics if metrics is not None else []
        self.pretrained_weights = pretrained_weights
        self.activation_output = activation_output
        self.fcl_dropout = fcl_dropout
        self.n_meta_variables = n_meta_variables
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assemble architecture parameters
        arch_paras = {"channels": channels, "pretrained_weights": pretrained_weights}
        if input_resolution is not None:
            arch_paras["input_resolution"] = input_resolution
        # Assemble classifier parameters
        classifier_paras = {
            "n_labels": n_labels,
            "fcl_dropout": fcl_dropout,
        }
        if n_meta_variables is not None:
            classifier_paras["n_meta_variables"] = n_meta_variables
        # Initialize architecture if None provided
        if architecture is None:
            self.architecture = architecture_dict["2D.Vanilla"](**arch_paras)
            self.arch_standardize = "z-score"
        # Initialize passed architecture from aucmedi library
        elif isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
            self.arch_standardize = supported_standardize_mode[architecture]
        # Initialize passed architecture as parameter
        else:
            self.architecture = architecture
            self.arch_standardize = None
        # Use "z-score" standarization as Fallback if the number of channels is not 3 (e.g. for non-RGB images)
        if (self.architecture.input_shape[-1] != 3) and (
            self.arch_standardize in ["torch", "tf", "caffe"]
        ):
            print(
                f"Warning: The architecture {self.architecture.__class__.__name__} is designed for 3-channel RGB images. Your input has {self.architecture.input_shape[-1]} channels. The recommended standardization mode for this architecture is '{self.arch_standardize}', which is not suitable for non-RGB images. Therefore, the standardization mode will be set to 'z-score' as a fallback."
            )
            self.arch_standardize = "z-score"

        # Obtain final input shape
        self.input_shape = self.architecture.input_shape  # e.g. (224, 224, 3)
        self.arch_resolution = self.architecture.input_shape[
            :-1
        ]  # e.g. (224, 224) -> for DataGenerator

        # Build model utilizing the selected architecture
        self.model_base = self.architecture.create_model()
        # output_shape() returns (height, width, channels) for 2D or (depth, height, width, channels) for 3D
        arch_output_shape = self.architecture.get_output_shape()

        # Initialize classifier
        classifier = Classifier(**classifier_paras)
        # Add classification head via Classifier
        self.model = classifier.build(
            model_base=self.model_base, arch_output_shape=arch_output_shape
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # Cache starting weights
        self.initialization_weights = [p.data.clone() for p in self.model.parameters()]

    # ---------------------------------------------#
    #                  Training                   #
    # ---------------------------------------------#

    def train_distributed(
        self,
        generator_fn,
        validation_generator_fn=None,
        world_size=None,
        iterations=None,
        epochs=20,
        learning_rate=0.0001,
        transfer_learning=False,
        transfer_epochs=10,
        fine_tuning_lr=None,
        callbacks=[],
        scheduler=None,
        class_weights=None,
    ):
        """Distributed (multi-GPU, single-node) counterpart of [train()][aucmedi.neural_network.model.NeuralNetwork.train].

        Spawns `world_size` processes (one per GPU) via `torch.multiprocessing.spawn`, each
        running DistributedDataParallel (DDP) on its own device. `train()` itself is untouched
        and remains the entry point for single-GPU/CPU training.

        ??? warning "generator_fn must be a factory, not a generator instance"
            Because a `BatchGenerator`/`WrapperLoader` holds process-local resources
            (`ThreadPool`s, open file handles) that cannot be pickled across process
            boundaries, each worker process must build its own generator. Pass a callable
            `generator_fn(rank, world_size) -> WrapperLoader/BatchGenerator/DataLoader` that
            constructs and returns a generator over **this rank's shard of the data**, e.g.:

            ```python
            def make_train_loader(rank, world_size):
                return create_batch_loader(samples[rank::world_size], "images_dir/",
                                           labels=class_ohe[rank::world_size],
                                           resize=model.arch_resolution,
                                           standardize_mode=model.arch_standardize)

            model.train_distributed(make_train_loader, epochs=50)
            ```

        ??? info "What happens to the trained weights"
            Only rank 0 writes the trained weights (and history) to a temporary checkpoint
            after training; the parent process loads that checkpoint back into `self.model`
            once all workers have joined. Per-epoch loss is averaged across ranks via
            `all_reduce` so it reflects the whole dataset, not one rank's shard. Logging and
            callbacks run on rank 0 only; the early-stopping decision is broadcast to all
            ranks so they stop in lockstep.

        Args:
            generator_fn (Callable[[int, int], WrapperLoader or BatchGenerator or DataLoader]):
                                                    Factory building the training generator for a given
                                                    `(rank, world_size)`; called once inside each worker process.
            validation_generator_fn (Callable[[int, int], WrapperLoader or BatchGenerator or DataLoader]):
                                                    Optional factory building the validation generator (same contract).
            world_size (int):                       Number of processes/GPUs to use. Defaults to `torch.cuda.device_count()`.
                                                    If greater than the number of visible GPUs, ranks share
                                                    devices (`gloo` backend) purely to smoke-test the distributed
                                                    code path on a single GPU -- not a real multi-GPU speedup.
            iterations (int):                       Number of batches per epoch. Ignored for generic DataLoaders
                                                    (those without a `set_length` method); use `None` to iterate
                                                    over all batches.
            epochs (int):                           Total number of epochs (includes transfer-learning epochs).
            learning_rate (float):                  Learning rate passed to the Adam optimizer.
            transfer_learning (bool):               If True, run a two-phase transfer-learning process.
            transfer_epochs (int):                  Epochs in the frozen phase. Must be less than `epochs`.
            fine_tuning_lr (float):                 Learning rate for the fine-tuning phase. Defaults to 0.1 × `learning_rate`.
            callbacks (list of Callback):           Custom Callback instances. Must be picklable, since they are
                                                    sent to every worker process.
            scheduler (torch.optim.lr_scheduler):   LR scheduler class (not instance) to initialize. `None` disables scheduling.
            class_weights (dict or list):           Per-class weights to handle class imbalance.

        Returns:
            history (dict):                         Training history with loss and metric logs per epoch.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "train_distributed() requires CUDA GPUs. Use train() for CPU or single-GPU training."
            )
        num_gpus = torch.cuda.device_count()
        if world_size is None:
            world_size = num_gpus
        if world_size < 2:
            raise ValueError(
                f"train_distributed() requires world_size >= 2 (got {world_size}). "
                "Use train() instead for single-GPU training."
            )
        if world_size > num_gpus:
            print(
                f"Warning: world_size={world_size} exceeds the {num_gpus} visible GPU(s); "
                f"ranks will share GPUs (device = rank % {num_gpus}) over the 'gloo' backend "
                "instead of 'nccl'. This only exercises the distributed code path locally for "
                "testing -- it will not run faster than train() and is not a supported way to "
                "do real multi-GPU training."
            )

        # Move the model (and cached init weights) to CPU before spawning: each worker
        # moves its own copy onto the GPU it owns (cuda:{rank}) instead of relying on
        # CUDA IPC sharing across processes, which only behaves cleanly when the
        # source and target device already match.
        original_device = self.device
        self.model.to("cpu")
        self.initialization_weights = [p.to("cpu") for p in self.initialization_weights]

        checkpoint_fd, checkpoint_path = tempfile.mkstemp(suffix=".pt")
        os.close(checkpoint_fd)
        try:
            mp.spawn(
                self._distributed_worker,
                args=(
                    world_size,
                    num_gpus,
                    generator_fn,
                    validation_generator_fn,
                    iterations,
                    epochs,
                    learning_rate,
                    transfer_learning,
                    transfer_epochs,
                    fine_tuning_lr,
                    callbacks,
                    scheduler,
                    class_weights,
                    checkpoint_path,
                ),
                nprocs=world_size,
                join=True,
            )
            # weights_only=False: this checkpoint is our own temp file, not an
            # untrusted third-party model, and it holds a plain history dict
            # alongside the state_dict (not tensors only).
            checkpoint = torch.load(
                checkpoint_path, map_location=original_device, weights_only=False
            )
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        # Restore the trained weights into this (parent-process) model instance
        self.device = original_device
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.initialization_weights = [
            p.to(self.device) for p in self.initialization_weights
        ]
        return checkpoint["history"]

    def _distributed_worker(
        self,
        rank,
        world_size,
        num_gpus,
        generator_fn,
        validation_generator_fn,
        iterations,
        epochs,
        learning_rate,
        transfer_learning,
        transfer_epochs,
        fine_tuning_lr,
        callbacks,
        scheduler,
        class_weights,
        checkpoint_path,
    ):
        """Per-process entry point spawned by `train_distributed()`. Not meant to be called directly."""
        # Map rank -> physical GPU. When world_size > num_gpus (local simulation),
        # multiple ranks share a device and must use "gloo" instead of "nccl".
        device_id = rank % num_gpus
        backend = "nccl" if world_size <= num_gpus else "gloo"
        ddp_setup(rank, world_size, device_id, backend=backend)
        self.device = torch.device(f"cuda:{device_id}")
        # transfer_learning freezes/unfreezes parameters via requires_grad between
        # phases, so the set of parameters that actually receive gradients changes
        # across iterations. Without find_unused_parameters=True, DDP's reducer
        # (built assuming every parameter participates every iteration) errors as
        # soon as frozen ones stop producing gradients.
        self.model = DDP(
            self.model.to(self.device),
            device_ids=[device_id],
            find_unused_parameters=transfer_learning,
        )

        training_generator = generator_fn(rank, world_size)
        validation_generator = (
            validation_generator_fn(rank, world_size)
            if validation_generator_fn is not None
            else None
        )

        history = self._fit(
            training_generator,
            validation_generator,
            iterations,
            epochs,
            learning_rate,
            transfer_learning,
            transfer_epochs,
            fine_tuning_lr,
            callbacks,
            scheduler,
            class_weights,
            rank=rank,
            world_size=world_size,
        )

        if rank == 0:
            torch.save(
                {"state_dict": self.model.module.state_dict(), "history": history},
                checkpoint_path,
            )

        destroy_process_group()

    # Training the Neural Network model
    def train(
        self,
        training_generator,
        validation_generator=None,
        iterations=None,
        epochs=20,
        learning_rate=0.0001,
        transfer_learning=False,
        transfer_epochs=10,
        fine_tuning_lr=None,
        callbacks=[],
        scheduler=None,
        class_weights=None,
    ):
        """Fitting function for the Neural Network model performing a training process.

        Accepts a `WrapperLoader` (standard AUCMEDI entry point via `create_batch_loader`),
        a raw `BatchGenerator`, or any generic PyTorch `DataLoader` whose `dataset` and
        loader instance expose `has_labels`, `has_metadata`, and `has_sample_weights`
        boolean attributes.

        If a validation generator is provided, validation loss is computed after each epoch.

        The transfer learning training runs two fitting passes: first with frozen base
        layers at `learning_rate`, then with all layers unfrozen at `fine_tuning_lr`.

        For multi-GPU data-parallel training, see
        [train_distributed()][aucmedi.neural_network.model.NeuralNetwork.train_distributed] instead.

        ??? info "History for Transfer Learning"
            Two history dicts are merged and returned as one. Keys are prefixed:

            ```
            tl_*  — transfer-learning phase (frozen layers)
            ft_*  — fine-tuning phase (unfrozen layers)
            ```

        Args:
            training_generator (WrapperLoader or BatchGenerator or DataLoader):
                                                    Generator used for training. Generic PyTorch DataLoaders
                                                    must expose `has_labels`, `has_metadata`, and
                                                    `has_sample_weights` on both the loader and its `dataset`.
            validation_generator (WrapperLoader or BatchGenerator or DataLoader):
                                                    Optional generator used for validation (same contract as above).
            iterations (int):                       Number of batches per epoch. Ignored for generic DataLoaders
                                                    (those without a `set_length` method); use `None` to iterate
                                                    over all batches.
            epochs (int):                           Total number of epochs (includes transfer-learning epochs).
            learning_rate (float):                  Learning rate passed to the Adam optimizer.
            transfer_learning (bool):               If True, run a two-phase transfer-learning process.
            transfer_epochs (int):                  Epochs in the frozen phase. Must be less than `epochs`.
            fine_tuning_lr (float):                 Learning rate for the fine-tuning phase. Defaults to 0.1 × `learning_rate`.
            callbacks (list of Callback):           Custom Callback instances (e.g. `ModelCheckpoint`, `MinEpochEarlyStopping`).
            scheduler (torch.optim.lr_scheduler):   LR scheduler class (not instance) to initialize. `None` disables scheduling.
            class_weights (dict or list):           Per-class weights to handle class imbalance.

        Returns:
            history (dict):                         Training history with loss and metric logs per epoch.
        """
        return self._fit(
            training_generator,
            validation_generator,
            iterations,
            epochs,
            learning_rate,
            transfer_learning,
            transfer_epochs,
            fine_tuning_lr,
            callbacks,
            scheduler,
            class_weights,
            rank=0,
            world_size=1,
        )

    def _fit(
        self,
        training_generator,
        validation_generator,
        iterations,
        epochs,
        learning_rate,
        transfer_learning,
        transfer_epochs,
        fine_tuning_lr,
        callbacks,
        scheduler,
        class_weights,
        rank=0,
        world_size=1,
    ):
        """Internal fitting routine shared by `train()` (rank=0, world_size=1) and the
        per-process worker spawned by `train_distributed()`."""
        if fine_tuning_lr is None:
            fine_tuning_lr = 0.1 * learning_rate

        early_stopping_callback = None
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise ValueError(
                    f"All callbacks must be instances of the Callback class. Found {type(callback)}."
                )
            if isinstance(callback, EarlyStoppingCallback):
                if early_stopping_callback is None:
                    early_stopping_callback = callback
                else:
                    # If multiple EarlyStoppingCallbacks are found, raise a warning since only the first one will be used
                    raise ValueError(f"Multiple EarlyStoppingCallbacks found. Using the first one: {early_stopping_callback}.")

        if transfer_learning and transfer_epochs >= epochs:
            if rank == 0:
                print(
                    "transfer_epochs should be lower than epochs when using transfer_learning. "
                    f"Received transfer_epochs={transfer_epochs}, epochs={epochs}. "
                    "Setting transfer_epochs to epochs - 1."
                )
            transfer_epochs = max(0, epochs - 1)

        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.lr_scheduler = None
        self.lr_scheduler_with_fb = None
        if scheduler is not None:
            # Check if scheduler is a ReduceLROnPlateau which requires feedback
            if scheduler == lr_scheduler.ReduceLROnPlateau:
                # For ReduceLROnPlateau, we need to pass feedback to the scheduler step function
                self.lr_scheduler_with_fb = scheduler(self.optimizer)
            else:
                # Initialize learning rate scheduler
                self.lr_scheduler = scheduler(self.optimizer)

        # Adjust number of iterations in training DataGenerator to allow repetition
        if iterations is not None and hasattr(training_generator, "set_length"):
            training_generator.set_length(iterations)

        ### Running a STANDARD training process
        if not transfer_learning:
            history_out = self._train_epoch(
                training_generator,
                validation_generator,
                epochs,
                iterations,
                class_weights,
                callbacks=callbacks,
                early_stopping_callback=early_stopping_callback,
                rank=rank,
                world_size=world_size,
            )

        ### Running a TRANSFER LEARNING training process
        else:
            # Freeze base model layers
            for name, param in self.model.named_parameters():
                if "avg_pool" not in name and "head" not in name:
                    param.requires_grad = False

            # Set high learning rate for initial training
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate,
            )
            if scheduler is not None:
                # Check if scheduler is a ReduceLROnPlateau which requires feedback
                if scheduler == lr_scheduler.ReduceLROnPlateau:
                    # For ReduceLROnPlateau, we need to pass feedback to the scheduler step function
                    self.lr_scheduler_with_fb = scheduler(self.optimizer)
                else:
                    # Initialize learning rate scheduler
                    self.lr_scheduler = scheduler(self.optimizer)
            # Run first training with frozen layers
            history_start = self._train_epoch(
                training_generator,
                validation_generator,
                transfer_epochs,
                iterations,
                class_weights,
                callbacks=callbacks,
                early_stopping_callback=early_stopping_callback,
                rank=rank,
                world_size=world_size,
            )

            # Unfreeze base model layers again
            for param in self.model.parameters():
                param.requires_grad = True

            # Set lower learning rate for fine-tuning
            self.optimizer = Adam(self.model.parameters(), lr=fine_tuning_lr)
            if scheduler is not None:
                if scheduler == lr_scheduler.ReduceLROnPlateau:
                    # For ReduceLROnPlateau, we need to pass feedback to the scheduler step function
                    self.lr_scheduler_with_fb = scheduler(self.optimizer)
                else:
                    # Initialize learning rate scheduler
                    self.lr_scheduler = scheduler(self.optimizer)
            ft_epochs = epochs - transfer_epochs

            # Run second training with unfrozen layers
            history_end = self._train_epoch(
                training_generator,
                validation_generator,
                ft_epochs,
                iterations,
                class_weights,
                callbacks=callbacks,
                early_stopping_callback=early_stopping_callback,
                rank=rank,
                world_size=world_size,
            )
            # Combine history dictionaries
            hs = {"tl_" + k: v for k, v in history_start.items()}
            he = {"ft_" + k: v for k, v in history_end.items()}
            history_out = {**hs, **he}

        # Reset number of iterations of the training DataGenerator
        if iterations is not None and hasattr(training_generator, "reset_length"):
            training_generator.reset_length()
        # Return fitting history
        return history_out

    def _reduce_average(self, total, count, world_size):
        """Average a (sum, count) pair across ranks via all_reduce.

        Ensures reported loss reflects the whole dataset instead of just one
        rank's shard; a no-op reduction to `total / count` when world_size <= 1.
        """
        if world_size <= 1:
            return total / count
        stats = torch.tensor([total, float(count)], device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (stats[0] / stats[1]).item()

    def _train_epoch(
        self,
        training_generator,
        validation_generator,
        epochs,
        iterations,
        class_weights,
        callbacks=[],
        early_stopping_callback=None,
        rank=0,
        world_size=1,
    ):
        """Internal function for training for a number of epochs.

        When `world_size > 1`, per-rank losses are averaged across ranks (see
        `_reduce_average`), logging/callbacks only run on rank 0, and the
        early-stopping decision is broadcast from rank 0 so all ranks stop in
        lockstep (a rank that stops alone would otherwise hang the others on
        the next collective all_reduce/broadcast call).
        """
        history = {
            "loss": [],
            "val_loss": [],
            "epoch_time": [],
            "learning_rate": [],
        }
        # Check that generator is a torch DataLoader or torch DataSet
        if not isinstance(training_generator, DataLoader):
            raise ValueError(
                "training_generator must be an instance of torch.utils.data.DataLoader"
            )

        # Cache generator flags to avoid repeated getattr() calls in hot loop
        train_set = training_generator.dataset
        train_has_labels = getattr(train_set, "has_labels", True)
        train_has_metadata = getattr(train_set, "has_metadata", False)
        train_has_sample_weights = getattr(train_set, "has_sample_weights", False)

        val_has_labels = None
        val_has_metadata = None
        val_has_sample_weights = None
        if validation_generator is not None:
            # Check that generator is a torch DataLoader
            if not isinstance(validation_generator, DataLoader):
                raise ValueError(
                    "validation_generator must be an instance of torch.utils.data.DataLoader"
                )
            val_set = validation_generator.dataset
            val_has_labels = getattr(val_set, "has_labels", True)
            val_has_metadata = getattr(val_set, "has_metadata", False)
            val_has_sample_weights = getattr(val_set, "has_sample_weights", False)

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            # Training loop
            self.model.train(True)
            self.epoch_start_time = time()
            for batch_idx, train_gen_output in enumerate(training_generator):
                if iterations is not None and batch_idx >= iterations:
                    break
                # Unravel generator output
                x, y, metadata, sample_weights = self.unravel_generator_output(
                    train_gen_output,
                    train_has_labels,
                    train_has_metadata,
                    train_has_sample_weights,
                )
                self.optimizer.zero_grad()
                outputs = self.model(x, metadata)
                batch_loss = self.loss(outputs, y)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss.item()
                batch_count += 1

            avg_loss = self._reduce_average(epoch_loss, batch_count, world_size)
            history["loss"].append(avg_loss)

            # Validation loop
            avg_val_loss = None
            if validation_generator is not None:
                val_loss = 0.0
                val_batch_count = 0
                self.model.train(False)

                with torch.no_grad():
                    for val_gen_output in validation_generator:
                        # Unravel generator output
                        x_val, y_val, metadata_val, _ = self.unravel_generator_output(
                            val_gen_output,
                            val_has_labels,
                            val_has_metadata,
                            val_has_sample_weights,
                        )
                        val_outputs = self.model(x_val, metadata_val)
                        val_batch_loss = self.loss(val_outputs, y_val)
                        val_loss += val_batch_loss.item()
                        val_batch_count += 1

                avg_val_loss = self._reduce_average(val_loss, val_batch_count, world_size)
                history["val_loss"].append(avg_val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            # Update learning rate scheduler if provided
            if self.lr_scheduler_with_fb is not None and avg_val_loss is not None:
                self.lr_scheduler_with_fb.step(avg_val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            ELAPSED_TIME = time() - self.epoch_start_time
            history["epoch_time"].append(ELAPSED_TIME)
            history["learning_rate"].append(current_lr)
            if rank == 0 and self.verbose:
                if avg_val_loss is not None:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f},  Val Loss: {avg_val_loss:.4f}, Time: {ELAPSED_TIME:.2f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {ELAPSED_TIME:.2f}s"
                    )

            # Callbacks and early stopping only run on rank 0 to avoid duplicate
            # logs/checkpoints; the stop decision is then broadcast to all ranks.
            should_stop = False
            if rank == 0:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs=history, model=self)

                if (
                    early_stopping_callback is not None
                    and early_stopping_callback.on_epoch_end(
                        epoch, logs=history, model=self
                    )
                ):
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    should_stop = True

            if world_size > 1:
                stop_tensor = torch.tensor(
                    1 if should_stop else 0, device=self.device
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break
        return history

    def unravel_generator_output(
        self,
        generator_output,
        has_labels=True,
        has_metadata=False,
        has_sample_weights=False,
    ):
        """Internal function to unravel the output of a DataGenerator
        and permute its shape as necessary.

        Args:
            generator_output (tuple):    Output of a DataGenerator batch.
            has_labels (bool):            Flag indicating whether the generator output contains labels.
            has_metadata (bool):          Flag indicating whether the generator output contains metadata.
            has_sample_weights (bool):    Flag indicating whether the generator output contains sample weights.

        Returns:
            x (torch.Tensor):            Input data tensor.
            y (torch.Tensor):            Label data tensor.
            metadata (torch.Tensor or None):   Metadata tensor if present, else None.
            sample_weights (torch.Tensor or None): Sample weights tensor if present, else None.
        """
        # Unravel generator output
        if has_labels:
            if has_sample_weights:
                data, y, sample_weights = generator_output
                if sample_weights is not None and isinstance(
                    sample_weights, np.ndarray
                ):
                    sample_weights = torch.from_numpy(sample_weights).float()
                    sample_weights = sample_weights.to(self.device)
            else:
                data, y = generator_output
                sample_weights = None
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()
                y = y.to(self.device)
        else:
            data = generator_output
            y = None
            sample_weights = None

        # Handle metadata case where data is a tuple (x, metadata)
        if has_metadata:
            x, metadata = data
            if isinstance(metadata, np.ndarray):
                metadata = torch.from_numpy(metadata).float()
            metadata = metadata.to(self.device)
        else:
            x = data
            metadata = None

        # Convert to torch tensors if numpy arrays
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()
        x = x.to(self.device)

        return x, y, metadata, sample_weights

    # ---------------------------------------------#
    #                 Prediction                  #
    # ---------------------------------------------#
    def predict(self, prediction_generator):
        """Prediction function for the Neural Network model.

        Accepts the same generator types as `train()`: `WrapperLoader`, `BatchGenerator`,
        or a generic PyTorch `DataLoader` that exposes `has_labels`, `has_metadata`, and
        `has_sample_weights` directly on the loader instance (not only on `dataset`).

        Args:
            prediction_generator (WrapperLoader or BatchGenerator or DataLoader):
                                                    Generator used for inference.

        Returns:
            preds (numpy.ndarray):                  Predictions with shape (n_samples, n_labels).
        """
        self.model.eval()
        all_preds = []

        # Cache generator flags to avoid repeated getattr() calls in hot loop
        pred_has_labels = getattr(prediction_generator, "has_labels", True)
        pred_has_metadata = getattr(prediction_generator, "has_metadata", False)
        pred_has_sample_weights = getattr(
            prediction_generator, "has_sample_weights", False
        )

        with torch.no_grad():
            for gen_out in prediction_generator:
                # Unravel generator output
                x, _, metadata, _ = self.unravel_generator_output(
                    gen_out,
                    pred_has_labels,
                    pred_has_metadata,
                    pred_has_sample_weights,
                )
                outputs = self.model(x, metadata)
                if self.activation_output == "softmax":
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                elif self.activation_output == "sigmoid":
                    outputs = torch.sigmoid(outputs)
                else:
                    pass  # No activation applied, raw logits returned
                preds = outputs.cpu().numpy()
                all_preds.append(preds)

        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds

    # ---------------------------------------------#
    #               Model Management              #
    # ---------------------------------------------#
    # Re-initialize model weights
    def reset_weights(self):
        """Re-initialize weights of the neural network model.

        Useful for training multiple models with the same NeuralNetwork object.
        """
        for p, init_p in zip(self.model.parameters(), self.initialization_weights):
            p.data.copy_(init_p)

    # Dump model to file
    def dump(self, file_path):
        """Store model to disk.

        Recommended to utilize the file format ".pt" or ".pth".

        Args:
            file_path (str):    Path to store the model on disk.
        """
        torch.save(self.model.state_dict(), file_path)

    # Load model from file
    def load(self, file_path):
        """Load neural network model and its weights from a file.

        Args:
            file_path (str):    Input path, from which the model will be loaded.
        """
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
