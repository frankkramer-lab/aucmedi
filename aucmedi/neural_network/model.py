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

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Internal libraries/scripts
from aucmedi.neural_network.architectures import (
    architecture_dict,
    supported_standardize_mode,
    Classifier,
)


# -----------------------------------------------------#
#            Neural Network (model) class             #
# -----------------------------------------------------#
# Class which represents the Neural Network
class NeuralNetwork:
    """ Neural Network class providing functionality for handling all model methods.

    This class is the third of the three pillars of AUCMEDI.

    ??? info "Pillars of AUCMEDI"
        - [aucmedi.data_processing.io_data.input_interface][]
        - [aucmedi.data_processing.data_generator.DataGenerator][]
        - [aucmedi.neural_network.model.NeuralNetwork][]

    With an initialized Neural Network model instance, it is possible to run training and predictions.

    ??? example "Example: How to use"
        ```python
        # Initialize model
        model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.ResNet50")
        # Do some training
        datagen_train = DataGenerator(samples[:100], "images_dir/", labels=class_ohe[:100],
                                      resize=model.arch_resolution, standardize_mode=model.arch_standardize)
        model.train(datagen_train, epochs=50)
        # Do some predictions
        datagen_test = DataGenerator(samples[100:150], "images_dir/", labels=None,
                                     resize=model.arch_resolution, standardize_mode=model.arch_standardize)
        preds = model.predict(datagen_test)
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

    ??? example "Example: How to obtain required parameters for the DataGenerator?"
        Be aware that the input_size and standardize_mode are just recommendations and
        can be changed by desire. <br>
        However, the recommended parameter are required for transfer learning.

        ```python title="Recommended way"
        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")

        my_dg = DataGenerator(samples, "images_dir/", labels=None,
                              resize=my_model.arch_resolution,                  # (224,224)
                              standardize_mode=my_model.arch_standardize)       # "torch"
        ```

        ```python title="Manual way"
        from aucmedi.neural_network.architectures import Classifier, \
                                                         architecture_dict, \
                                                         supported_standardize_mode

        my_arch = architecture_dict["3D.DenseNet121"](n_labels=4,
                                                      channels=1,
                                                      input_resolution=(128,128,128))

        my_model = NeuralNetwork(n_labels=None, channels=None, architecture=my_arch)

        from aucmedi.neural_network.architectures import supported_standardize_mode
        sf_norm = supported_standardize_mode["3D.DenseNet121"]
        my_dg = DataGenerator(samples, "images_dir/", labels=None,
                              resize=(128,128,128),                        # (128,128,128)
                              standardize_mode=sf_norm)                    # "torch"
        ```

    ??? example "Example: How to integrate metadata in AUCMEDI?"
        ```python
        from aucmedi import *
        import numpy as np

        my_metadata = np.random.rand(len(samples), 10)

        my_model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121",
                                  n_meta_variables=10)

        my_dg = DataGenerator(samples, "images_dir/",
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
        learning_rate=0.0001,
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
            activation_output (str):                Activation function which should be used in the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            fcl_dropout (bool):                     Option whether to utilize an additional Linear & Dropout layer in the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            n_meta_variables (int):                   Number of metadata variables, which should be included in the classification head.
                                                    If `None`is provided, no metadata integration block will be added to the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
            verbose (int):                          Option (0/1) how much information should be written to stdout.

        ???+ danger
            Class attributes can be modified also after initialization, at will.
            However, be aware of unexpected adverse effects (experimental)!

        Attributes:
            tf_epochs (int, default=5):             Transfer Learning configuration: Number of epochs with frozen layers except classification head.
            tf_lr_start (float, default=1e-4):      Transfer Learning configuration: Starting learning rate for frozen layer fitting.
            tf_lr_end (float, default=1e-5):        Transfer Learning configuration: Starting learning rate after layer unfreezing.
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
        self.learning_rate = learning_rate
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

        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        # Cache starting weights
        self.initialization_weights = [p.data.clone() for p in self.model.parameters()]

    # ---------------------------------------------#
    #               Class Variables               #
    # ---------------------------------------------#
    # Transfer Learning configurations
    # TODO: train params
    tf_epochs = 10
    tf_lr_start = 1e-4
    tf_lr_end = 1e-5

    # ---------------------------------------------#
    #                  Training                   #
    # ---------------------------------------------#

    # Training the Neural Network model
    def train(
        self,
        training_generator,
        validation_generator=None,
        epochs=20,
        iterations=None,
        callbacks=[],
        early_stopping_callback=None,
        class_weights=None,
        transfer_learning=False,
    ):
        """Fitting function for the Neural Network model performing a training process.

        It is also possible to pass custom Callback classes in order to obtain more information.

        If an optional validation [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator]
        is provided, a validation set is analyzed regularly during the training process (after each epoch).

        The transfer learning training runs two fitting processes.
        The first one with frozen base model layers and a high learning rate,
        whereas the second one with unfrozen layers and a small learning rate.

        ??? info "History for Transfer Learning"
            For the transfer learning training, two history dictionaries will be created.

            However, in order to provide consistency with the single training without transfer learning,
            only a single history dictionary will be returned.

            For differentiation prefixes are added in front of the corresponding logging keys:
            ```
            - History Start ->  prefix : tl     for "transfer learning"
            - History End   ->  prefix : ft     for "fine tuning"
            ```

        Args:
            training_generator (DataGenerator):     A data generator which will be used for training.
            validation_generator (DataGenerator):   A data generator which will be used for validation.
            epochs (int):                           Number of epochs. A single epoch is defined as one iteration through
                                                    the complete data set.
            iterations (int):                       Number of iterations (batches) in a single epoch. If None is provided, the number of iterations is determined by the length of the training_generator.
            callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
            class_weights (dictionary or list):     A list or dictionary of float values to handle class unbalance.
            transfer_learning (bool):               Option whether a transfer learning training should be performed. If true, a minimum of 10 epochs will be trained.

        Returns:
            history (dict):                   A history dictionary which contains several logs.
        """
        # Adjust number of iterations in training DataGenerator to allow repitition
        if iterations is not None:
            training_generator.set_length(iterations)
        # Running a standard training process
        if not transfer_learning:
            history_out = self._train_epoch(
                training_generator,
                validation_generator,
                epochs,
                iterations,
                class_weights,
                callbacks=callbacks,
                early_stopping_callback=early_stopping_callback,
            )
        # Running a transfer learning training process
        else:
            # Freeze base model layers
            for name, param in self.model.named_parameters():
                if "avg_pool" not in name and "head" not in name:
                    param.requires_grad = False

            # Set high learning rate for initial training
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.tf_lr_start,
            )

            # Run first training with frozen layers
            history_start = self._train_epoch(
                training_generator,
                validation_generator,
                self.tf_epochs,
                iterations,
                class_weights,
                callbacks=callbacks,
                early_stopping_callback=early_stopping_callback,
            )

            # Unfreeze base model layers again
            for param in self.model.parameters():
                param.requires_grad = True

            # Set lower learning rate for fine-tuning
            self.optimizer = Adam(self.model.parameters(), lr=self.tf_lr_end)

            # Run second training with unfrozen layers
            history_end = self._train_epoch(
                training_generator,
                validation_generator,
                epochs,
                iterations,
                class_weights,
                callbacks=callbacks,
                early_stopping_callback=early_stopping_callback,
            )

            # Combine history dictionaries
            hs = {"tl_" + k: v for k, v in history_start.items()}
            he = {"ft_" + k: v for k, v in history_end.items()}
            history_out = {**hs, **he}

        # Reset number of iterations of the training DataGenerator
        if iterations is not None:
            training_generator.reset_length()
        # Return fitting history
        return history_out

    def _train_epoch(
        self,
        training_generator,
        validation_generator,
        epochs,
        iterations,
        class_weights,
        callbacks=[],
        early_stopping_callback=None,
    ):
        """Internal function for training for a number of epochs."""
        history = {
            "loss": [],
            "val_loss": [],
            "epoch_time": [],
        }
        # Check that generator is a torch DataLoader
        if not isinstance(training_generator, DataLoader):
            raise ValueError(
                "training_generator must be an instance of torch.utils.data.DataLoader"
            )

        # Cache generator flags to avoid repeated getattr() calls in hot loop
        train_has_labels = getattr(training_generator, "has_labels", True)
        train_has_metadata = getattr(training_generator, "has_metadata", False)
        train_has_sample_weights = getattr(
            training_generator, "has_sample_weights", False
        )

        val_has_labels = None
        val_has_metadata = None
        val_has_sample_weights = None
        if validation_generator is not None:
            # Check that generator is a torch DataLoader
            if not isinstance(validation_generator, DataLoader):
                raise ValueError(
                    "validation_generator must be an instance of torch.utils.data.DataLoader"
                )
            val_has_labels = getattr(validation_generator, "has_labels", True)
            val_has_metadata = getattr(validation_generator, "has_metadata", False)
            val_has_sample_weights = getattr(
                validation_generator, "has_sample_weights", False
            )

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

            avg_loss = epoch_loss / batch_count
            history["loss"].append(avg_loss)

            # Validation loop
            avg_val_loss = None  # Provide smth for print-out
            if validation_generator is not None:
                val_loss = 0.0
                val_batch_count = 0
                self.model.train(False)

                with torch.no_grad():
                    for val_gen_output in validation_generator:
                        # Unravel generator output
                        x_val, y_val, metadata_val, sample_weights_val = (
                            self.unravel_generator_output(
                                val_gen_output,
                                val_has_labels,
                                val_has_metadata,
                                val_has_sample_weights,
                            )
                        )

                        val_outputs = self.model(x_val, metadata_val)
                        val_batch_loss = self.loss(val_outputs, y_val)

                        val_loss += val_batch_loss.item()
                        val_batch_count += 1

                avg_val_loss = val_loss / val_batch_count
                history["val_loss"].append(avg_val_loss)
            ELAPSED_TIME = time() - self.epoch_start_time
            history["epoch_time"].append(ELAPSED_TIME)
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f},  Val Loss: {avg_val_loss:.4f}, Time: {ELAPSED_TIME:.2f}s"
                )
            for callback in callbacks:
                print(callback.on_epoch_end(epoch, logs=history))
            if (
                early_stopping_callback is not None
                and early_stopping_callback.on_epoch_end(epoch, logs=history)
            ):
                print(f"Early stopping triggered at epoch {epoch + 1}.")
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
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        else:
            x = x.float()

        x = x.to(self.device)
        y = y.to(self.device)

        # Convert from NHWC/NDHWC format to NCHW/NCDHW format for PyTorch
        if (
            x.dim() == 4
        ):  # 2D: (batch, height, width, channels) -> (batch, channels, height, width)
            x = x.permute(0, 3, 1, 2)
        elif (
            x.dim() == 5
        ):  # 3D: (batch, depth, height, width, channels) -> (batch, channels, depth, height, width)
            x = x.permute(0, 4, 1, 2, 3)

        return x, y, metadata, sample_weights

    # ---------------------------------------------#
    #                 Prediction                  #
    # ---------------------------------------------#
    def predict(self, prediction_generator):
        """Prediction function for the Neural Network model.

        The fitted model will predict classifications for the provided [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            prediction_generator (DataGenerator):   A data generator which will be used for inference.

        Returns:
            preds (numpy.ndarray):                  A NumPy array of predictions formatted with shape (n_samples, n_labels).
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
