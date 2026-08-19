# ==============================================================================#
#  Author:       Dominik Müller                                                #
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import os
import signal
import tempfile
from queue import Empty
from aucmedi.utils.callbacks import ModelCheckpoint, CSVLogger
from pathos.helpers import mp  # instead of 'import multiprocessing as mp'
import numpy as np
import shutil
from torch.utils.data import DataLoader, RandomSampler

# Internal libraries
from aucmedi import NeuralNetwork
from aucmedi.data_processing.wrapper_loader import create_batch_loader, WrapperLoader
from aucmedi.sampling import sampling_split, sampling_kfold
from aucmedi.ensemble.aggregate import aggregate_dict
from aucmedi.ensemble.metalearner import metalearner_dict
from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base
from aucmedi.ensemble.aggregate.agg_base import Aggregate_Base


# -----------------------------------------------------#
#              Generator Resolution Helper            #
# -----------------------------------------------------#
# Resolves the two generator wrapper styles Composite accepts -- a plain torch
# DataLoader wrapping a DataGenerator, or a WrapperLoader wrapping a
# BatchGenerator -- into (template_generator, num_workers). A bare/unwrapped
# DataGenerator or BatchGenerator is NOT accepted: num_workers lives on the
# wrapper (DataLoader/WrapperLoader), not on the underlying generator, and
# DataGenerator additionally has no batch_size/shuffle of its own (those are
# copied down from the DataLoader below) -- both are required downstream to
# rebuild a per-fold generator.
def __resolve_template_generator__(generator):
    if isinstance(generator, DataLoader):
        num_workers = generator.num_workers
        template_generator = generator.dataset
        template_generator.batch_size = generator.batch_size
        template_generator.shuffle = isinstance(generator.sampler, RandomSampler)
    elif isinstance(generator, WrapperLoader):
        num_workers = generator.num_workers
        template_generator = generator.batch_generator
    else:
        raise ValueError(
            "Invalid generator type: Must be a WrapperLoader or a torch "
            "DataLoader wrapping a DataGenerator!"
        )
    return template_generator, num_workers


# -----------------------------------------------------#
#            Ensemble Learning: Composite             #
# -----------------------------------------------------#
class Composite:
    """A Composite class providing functionality for cross-validation and metalearner based ensemble learning.

    The Composite strategy combines the homogeneous [Bagging][aucmedi.ensemble.Bagging] and the heterogeneous
    [Stacking][aucmedi.ensemble.Stacking] technique.

    If a metalearner is selected, a percentage sampling split is applied. For an aggregate function, this is not done.
    The remaining training data is sampled via a cross-validation. For each fold, a different model is trained
    returning into a heterogenous ensemble.
    Predictions for this heterogenous ensemble are combined with the fitted metalearner model or an aggregate function.

    Instead of utilizing the fixed parameters of the [DataGenerator][aucmedi.data_processing.data_generator],
    default paramters for Resizing and Standardize of the associated models are used (if `fixed_datagenerator=True`).

    ???+ example
        ```python
        # Initialize some NeuralNetwork models
        model_a = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet50")
        model_b = NeuralNetwork(n_labels=4, channels=3, architecture="2D.MobileNetV2")
        model_c = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB1")

        # Initialize Composite object
        el = Composite(model_list=[model_a, model_b, model_c],
                       metalearner="logistic_regression", k_fold=3)

        # Initialize training WrapperLoader for complete training data
        train_loader = create_batch_loader(samples_train, "images_dir/",
                                           labels=train_labels_ohe, batch_size=3,
                                           resize=model_a.arch_resolution,
                                           standardize_mode=model_a.arch_standardize)
        # Train neural network and metalearner models
        el.train(train_loader, epochs=100)

        # Initialize testing WrapperLoader for testing data
        test_loader = create_batch_loader(samples_test, "images_dir/",
                                          resize=model_a.arch_resolution,
                                          standardize_mode=model_a.arch_standardize)
        # Run Inference
        preds = el.predict(test_loader)
        ```

    !!! warning "Training Time Increase"
        Composite sequentially performs fitting processes for multiple models, which will drastically increase training time.

    ??? warning "Generator re-initialization"
        The passed generator for the train() and predict() function of the Composite class will be re-initialized!

        This can result in redundant image preparation if `prepare_images=True`.

        The `resize` and `standardize_mode` parameters are automatically overridden per model using
        `model.arch_resolution` and `model.arch_standardize` respectively.

        If desired (but not recommended!), these attributes can be set manually:
        ```python
        model_a = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet50",
                                input_resolution=(64, 64))
        ```

    ??? warning "NeuralNetwork re-initialization"
        The passed NeuralNetwork for the train() and predict() function of the Composite class will be re-initialized!

        Attention: Metrics are not passed to the processes due to pickling issues.

    ??? info "Technical Details"
        For the training and inference process, each model is trained in an individual subprocess via the Python multiprocessing package.

        This isolates GPU memory between models, ensuring PyTorch releases VRAM between successive training runs.
        The subprocess spawn method is used (`mp.set_start_method("spawn")`) for CUDA compatibility.
    """

    def __init__(
        self,
        model_list,
        metalearner="logistic_regression",
        k_fold=3,
        sampling=[0.85, 0.15],
        fixed_datagenerator=False,
    ):
        """Initialization function for creating a Composite object.

        Args:
            model_list (list of NeuralNetwork):         List of instances of AUCMEDI neural network class.
                                                        The number of models (`len(model_list)`) have to be equal to `k_fold`.
            metalearner (str, Metalearner or Aggregate):Metalearner class instance / a string for an AUCMEDI Metalearner,
                                                        or Aggregate function / a string for an AUCMEDI Aggregate function.
            k_fold (int):                               Number of folds (k) for the Cross-Validation. Must be at least 2.
            sampling (list of float):                   List of percentage values with split sizes. Should be 2x percentage values
                                                        for heterogenous metalearner (must sum up to 1.0).
            fixed_datagenerator (bool):                 Boolean, whether using fixed parameters of passed DataGenerator or
                                                        using default architecture paramters for Resizing and Standardize.
        """
        # Cache class variables
        self.model_list = model_list
        self.metalearner = metalearner
        self.sampling = sampling
        self.k_fold = k_fold
        self.fixed_datagenerator = fixed_datagenerator
        self.sampling_seed = 0
        self.cache_dir = None

        # Initialize Metalearner
        if isinstance(metalearner, str) and metalearner in metalearner_dict:
            self.ml_model = metalearner_dict[metalearner]()
        elif isinstance(metalearner, str) and metalearner in aggregate_dict:
            self.ml_model = aggregate_dict[metalearner]()
        elif isinstance(metalearner, Metalearner_Base) or isinstance(
            metalearner, Aggregate_Base
        ):
            self.ml_model = metalearner
        else:
            raise TypeError(
                "Unknown type of Metalearner (neither known "
                + "ensembler nor Aggregate or Metalearner class)!"
            )

        # Verify model list length
        if k_fold != len(model_list):
            raise ValueError("Length of model_list and k_fold has to be equal!")

        # Set multiprocessing method to spawn
        mp.set_start_method("spawn", force=True)

    def train(
        self,
        training_generator,
        epochs=20,
        iterations=None,
        callbacks=[],
        transfer_learning=False,
        learning_rate=0.0001,
        transfer_epochs=10,
        fine_tuning_lr=None,
        scheduler=None,
        metalearner_fitting=True,
    ):
        """Training function for fitting the provided NeuralNetwork models.

        The training data will be sampled according to a percentage split in which WrapperLoaders
        for model training and metalearner training are created if a metalearner is provided.
        Otherwise all data is used for model training. The model training subset is furthermore
        sampled via cross-validation.

        It is also possible to pass custom Callback classes in order to obtain more information.

        For more information on the fitting process, check out [NeuralNetwork.train()][aucmedi.neural_network.model.NeuralNetwork.train].

        Args:
            training_generator (WrapperLoader or DataLoader):     A generator which will be used for training (will be split according
                                                                    to percentage split and k-fold cross-validation sampling).
                                                                    Must be a WrapperLoader or a torch DataLoader wrapping a DataGenerator.
            epochs (int):                           Number of epochs. A single epoch is defined as one iteration through
                                                    the complete data set.
            iterations (int):                       Number of iterations (batches) in a single epoch.
            callbacks (list of Callback classes):   A list of Callback classes for custom evaluation (e.g. ModelCheckpoint).
            transfer_learning (bool):               Option whether a transfer learning training should be performed.
            learning_rate (float):                  Learning rate passed to the optimizer.
            transfer_epochs (int):                  Number of epochs used in the frozen transfer learning phase.
            fine_tuning_lr (float):                 Learning rate used during fine-tuning. Defaults to 0.1 * learning_rate.
            scheduler (torch.optim.lr_scheduler):   A PyTorch learning rate scheduler class to be initialized.
            metalearner_fitting (bool):             Option whether the Metalearner fitting process should be included in the
                                                    Composite training process. The `train_metalearner()` function can also be
                                                    run manually (or repeatedly).
        Returns:
            history (dict):                         A history dictionary which contains several logs.
        """
        # Resolve generator type (DataLoader or WrapperLoader). Keep the original
        # training_generator (a real DataLoader/WrapperLoader) untouched -- it is
        # passed as-is to train_metalearner() below, which resolves it again itself.
        template_generator, self.num_workers = __resolve_template_generator__(
            training_generator
        )

        history_composite = {}  # Final history dictionary

        # Create temporary model directory
        self.cache_dir = tempfile.TemporaryDirectory(
            prefix="aucmedi.tmp.", suffix=".composite"
        )

        # Obtain training data
        x = template_generator.samples
        y = template_generator.labels
        m = template_generator.metadata

        # Apply percentage split sampling for metalearner
        if isinstance(self.ml_model, Metalearner_Base):
            ps_sampling = sampling_split(
                x,
                y,
                m,
                sampling=self.sampling,
                stratified=True,
                iterative=True,
                seed=self.sampling_seed,
            )
            # Pack data according to sampling
            if len(ps_sampling[0]) == 3:
                x, y, m = ps_sampling[0]
            else:
                x, y = ps_sampling[0]

        # Apply cross-validaton sampling
        cv_sampling = sampling_kfold(
            x, y, m, n_splits=self.k_fold, stratified=True, iterative=True
        )

        # Sequentially iterate over model list
        for i in range(len(self.model_list)):
            # Pack data into a tuple
            fold = cv_sampling[i]
            if len(fold) == 4:
                train_x, train_y, test_x, test_y = fold
                data = (train_x, train_y, None, test_x, test_y, None)
            else:
                data = fold

            # Create model specific callback list
            callbacks_model = callbacks.copy()
            # Extend Callback list
            path_model = os.path.join(self.cache_dir.name, "cv_" + str(i) + ".model.pt")
            cb_mc = ModelCheckpoint(
                path_model,
                monitor="val_loss",
                mode="min",
            )
            cb_cl = CSVLogger(
                os.path.join(self.cache_dir.name, "cv_" + str(i) + ".logs.csv"),
                separator=",",
                append=True,
            )
            callbacks_model.extend([cb_mc, cb_cl])

            # Gather NeuralNetwork parameters
            model_paras = {
                "n_labels": self.model_list[i].n_labels,
                "channels": self.model_list[i].channels,
                "architecture": self.model_list[i].architecture,
                "pretrained_weights": self.model_list[i].pretrained_weights,
                "loss": self.model_list[i].loss,
                "metrics": None,
                "activation_output": self.model_list[i].activation_output,
                "fcl_dropout": self.model_list[i].fcl_dropout,
                "n_meta_variables": self.model_list[i].n_meta_variables,
            }

            # Gather DataGenerator parameters
            datagen_paras = {
                "path_imagedir": template_generator.path_imagedir,
                "batch_size": template_generator.batch_size,
                "data_aug": template_generator.data_aug,
                "seed": template_generator.seed,
                "subfunctions": template_generator.subfunctions,
                "shuffle": template_generator.shuffle,
                "standardize_mode": self.model_list[i].arch_standardize,
                "resize": self.model_list[i].arch_resolution,
                "grayscale": template_generator.grayscale,
                "two_dim": template_generator.two_dim,
                "prepare_images": template_generator.prepare_images,
                "sample_weights": template_generator.sample_weights,
                "image_format": template_generator.image_format,
                "loader": template_generator.sample_loader,
                "num_workers": self.num_workers,
                "kwargs": template_generator.kwargs,
            }

            # Gather training parameters
            parameters_training = {
                "epochs": epochs,
                "iterations": iterations,
                "callbacks": callbacks_model,
                "transfer_learning": transfer_learning,
                "learning_rate": learning_rate,
                "transfer_epochs": transfer_epochs,
                "fine_tuning_lr": fine_tuning_lr,
                "scheduler": scheduler,
            }

            # Start training process
            process_queue = mp.Queue()
            process_train = mp.Process(
                target=__training_process__,
                args=(
                    process_queue,
                    data,
                    model_paras,
                    datagen_paras,
                    parameters_training,
                ),
            )
            cv_history = __run_subprocess__(
                process_train, process_queue, label=f"Training (fold {i})"
            )
            # Combine logged history objects
            hnn = {"cv_" + str(i) + "." + k: v for k, v in cv_history.items()}
            history_composite = {**history_composite, **hnn}

        # Perform metalearner model training
        if isinstance(self.ml_model, Metalearner_Base):
            if metalearner_fitting:
                self.train_metalearner(training_generator)

        # Return Composite history object
        return history_composite

    def train_metalearner(self, training_generator):
        """Training function for fitting the Metalearner model.

        Function will be called automatically in the `train()` function if
        the parameter `metalearner_fitting` is true.

        However, this function can also be called multiple times for training
        different Metalearner types without the need of time-extensive
        re-training of the [NeuralNetwork][aucmedi.neural_network.model] models.

        Args:
            training_generator (WrapperLoader or DataLoader):     A generator which will be used for training (will be split according
                                                                    to percentage split). Must be a WrapperLoader or a torch DataLoader
                                                                    wrapping a DataGenerator.
        """
        # Skipping metalearner training if aggregate function
        if isinstance(self.ml_model, Aggregate_Base):
            return

        # Resolve generator type (DataLoader or WrapperLoader)
        training_generator, self.num_workers = __resolve_template_generator__(
            training_generator
        )

        preds_ensemble = []

        # Obtain training data
        x = training_generator.samples
        y = training_generator.labels
        m = training_generator.metadata

        # Apply percentage split sampling for metalearner
        if isinstance(self.ml_model, Metalearner_Base):
            ps_sampling = sampling_split(
                x,
                y,
                m,
                sampling=self.sampling,
                stratified=True,
                iterative=True,
                seed=self.sampling_seed,
            )
        # Pack data according to sampling
        if len(ps_sampling[0]) == 3:
            data_ensemble = ps_sampling[1]
        else:
            data_ensemble = (*ps_sampling[1], None)

        # Identify path to model directory
        if isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            path_model_dir = self.cache_dir.name
        else:
            path_model_dir = self.cache_dir

        # Sequentially iterate over model list
        for i in range(len(self.model_list)):
            # Load current model
            path_model = os.path.join(path_model_dir, "cv_" + str(i) + ".model.pt")

            # Gather NeuralNetwork parameters
            model_paras = {
                "n_labels": self.model_list[i].n_labels,
                "channels": self.model_list[i].channels,
                "input_resolution": self.model_list[i].arch_resolution,
                "architecture": self.model_list[i].architecture,
                "pretrained_weights": self.model_list[i].pretrained_weights,
                "loss": self.model_list[i].loss,
                "metrics": None,
                "activation_output": self.model_list[i].activation_output,
                "fcl_dropout": self.model_list[i].fcl_dropout,
                "n_meta_variables": self.model_list[i].n_meta_variables,
            }

            # Gather DataGenerator parameters
            datagen_paras = {
                "path_imagedir": training_generator.path_imagedir,
                "batch_size": training_generator.batch_size,
                "data_aug": training_generator.data_aug,
                "seed": training_generator.seed,
                "subfunctions": training_generator.subfunctions,
                "shuffle": training_generator.shuffle,
                "standardize_mode": self.model_list[i].arch_standardize,
                "resize": self.model_list[i].arch_resolution,
                "grayscale": training_generator.grayscale,
                "two_dim": training_generator.two_dim,
                "prepare_images": training_generator.prepare_images,
                "sample_weights": training_generator.sample_weights,
                "image_format": training_generator.image_format,
                "loader": training_generator.sample_loader,
                "num_workers": self.num_workers,
                "kwargs": training_generator.kwargs,
            }

            # Start inference process for model i
            process_queue = mp.Queue()
            process_pred = mp.Process(
                target=__prediction_process__,
                args=(
                    process_queue,
                    model_paras,
                    path_model,
                    data_ensemble,
                    datagen_paras,
                ),
            )
            preds = __run_subprocess__(
                process_pred, process_queue, label=f"Ensemble prediction (model {i})"
            )

            # Append preds to ensemble
            preds_ensemble.append(preds)

        # Preprocess prediction ensemble
        preds_ensemble = np.array(preds_ensemble)
        preds_ensemble = np.swapaxes(preds_ensemble, 0, 1)
        s, m, c = preds_ensemble.shape
        x_stack = np.reshape(preds_ensemble, (s, m * c))

        # Start training of stacked metalearner
        if isinstance(self.ml_model, Metalearner_Base):
            _, y_stack, _ = data_ensemble
            self.ml_model.train(x_stack, y_stack)
            # Store metalearner model to disk
            path_metalearner = os.path.join(path_model_dir, "metalearner.model.pickle")
            self.ml_model.dump(path_metalearner)

    def predict(self, prediction_generator, return_ensemble=False):
        """Prediction function for Composite.

        The fitted models and selected Metalearner/Aggregate function will predict classifications
        for the provided [WrapperLoader][aucmedi.data_processing.wrapper_loader.WrapperLoader].

        !!! info
            More about Metalearners can be found here: [Metelearner][aucmedi.ensemble.metalearner]

            More about Aggregate functions can be found here: [aggregate][aucmedi.ensemble.aggregate]

        Args:
            prediction_generator (WrapperLoader or DataLoader):   A generator which will be used for inference.
                                                                Must be a WrapperLoader or a torch DataLoader wrapping a DataGenerator.
            return_ensemble (bool):                 Option, whether gathered ensemble of predictions should be returned.

        Returns:
            preds (numpy.ndarray):                  A NumPy array of predictions formatted with shape (n_samples, n_labels).
            ensemble (numpy.ndarray):               Optional ensemble of predictions: Will be only passed if `return_ensemble=True`.
                                                    Shape (n_models, n_samples, n_labels).
        """
        # Verify if there is a linked cache dictionary
        con_tmp = isinstance(
            self.cache_dir, tempfile.TemporaryDirectory
        ) and os.path.exists(self.cache_dir.name)
        con_var = (
            self.cache_dir is not None
            and not isinstance(self.cache_dir, tempfile.TemporaryDirectory)
            and os.path.exists(self.cache_dir)
        )
        if not con_tmp and not con_var:
            raise FileNotFoundError(
                "Composite instance does not have a valid" + "model cache directory!"
            )

        # Resolve generator type (DataLoader or WrapperLoader)
        prediction_generator, self.num_workers = __resolve_template_generator__(
            prediction_generator
        )

        # Initialize some variables
        preds_ensemble = []
        preds_final = []

        # Extract data
        data_test = (prediction_generator.samples, prediction_generator.labels, prediction_generator.metadata)

        # Identify path to model directory
        if isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            path_model_dir = self.cache_dir.name
        else:
            path_model_dir = self.cache_dir

        # Sequentially iterate over model list
        for i in range(len(self.model_list)):
            path_model = os.path.join(path_model_dir, "cv_" + str(i) + ".model.pt")

            # Gather NeuralNetwork parameters
            model_paras = {
                "n_labels": self.model_list[i].n_labels,
                "channels": self.model_list[i].channels,
                "input_resolution": self.model_list[i].arch_resolution,
                "architecture": self.model_list[i].architecture,
                "pretrained_weights": self.model_list[i].pretrained_weights,
                "loss": self.model_list[i].loss,
                "metrics": None,
                "activation_output": self.model_list[i].activation_output,
                "fcl_dropout": self.model_list[i].fcl_dropout,
                "n_meta_variables": self.model_list[i].n_meta_variables,
            }

            # Gather DataGenerator parameters
            datagen_paras = {
                "path_imagedir": prediction_generator.path_imagedir,
                "batch_size": prediction_generator.batch_size,
                "data_aug": prediction_generator.data_aug,
                "seed": prediction_generator.seed,
                "subfunctions": prediction_generator.subfunctions,
                "shuffle": prediction_generator.shuffle,
                "standardize_mode": self.model_list[i].arch_standardize,
                "resize": self.model_list[i].arch_resolution,
                "grayscale": prediction_generator.grayscale,
                "two_dim": prediction_generator.two_dim,
                "prepare_images": prediction_generator.prepare_images,
                "sample_weights": prediction_generator.sample_weights,
                "image_format": prediction_generator.image_format,
                "loader": prediction_generator.sample_loader,
                "num_workers": self.num_workers,
                "kwargs": prediction_generator.kwargs,
            }

            # Start inference process for model i
            process_queue = mp.Queue()
            process_pred = mp.Process(
                target=__prediction_process__,
                args=(process_queue, model_paras, path_model, data_test, datagen_paras),
            )
            preds = __run_subprocess__(
                process_pred, process_queue, label=f"Prediction (model {i})"
            )

            # Append preds to ensemble
            preds_ensemble.append(preds)

        # Preprocess prediction ensemble
        preds_ensemble = np.array(preds_ensemble)
        preds_ensemble = np.swapaxes(preds_ensemble, 0, 1)

        # Apply heterogenous metalearner
        if isinstance(self.ml_model, Metalearner_Base):
            s, m, c = preds_ensemble.shape
            x_stack = np.reshape(preds_ensemble, (s, m * c))
            preds_final = self.ml_model.predict(data=x_stack)
        # Apply homogeneous aggregate function
        elif isinstance(self.ml_model, Aggregate_Base):
            for i in range(preds_ensemble.shape[0]):
                pred_sample = self.ml_model.aggregate(preds_ensemble[i, :, :])
                preds_final.append(pred_sample)

        # Convert prediction list to NumPy
        preds_final = np.asarray(preds_final)

        # Return ensembled predictions
        if return_ensemble:
            return preds_final, np.swapaxes(preds_ensemble, 1, 0)
        else:
            return preds_final

    # Dump model to file
    def dump(self, directory_path):
        """Store temporary Composite models directory permanently to disk at desired location.

        If the model directory is a provided path which is already persistent on the disk,
        the directory is copied in order to keep original data persistent.

        Args:
            directory_path (str):       Path to store the model directory on disk.
        """
        if self.cache_dir is None:
            raise FileNotFoundError(
                "Composite does not have a valid model cache directory!"
            )
        elif isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            shutil.copytree(self.cache_dir.name, directory_path, dirs_exist_ok=True)
            self.cache_dir.cleanup()
            self.cache_dir = directory_path
        else:
            shutil.copytree(self.cache_dir, directory_path, dirs_exist_ok=True)
            self.cache_dir = directory_path

    # Load model from file
    def load(self, directory_path):
        """Load a Composite model directory which can be used for Metalearner based inference.

        Args:
            directory_path (str):       Input path, from which the Composite models will be loaded.
        """
        # Check directory existence
        if not os.path.exists(directory_path):
            raise FileNotFoundError(
                "Provided model directory path does not exist!", directory_path
            )
        # Check model existence
        for i in range(len(self.model_list)):
            path_model = os.path.join(directory_path, "cv_" + str(i) + ".model.pt")
            if not os.path.exists(path_model):
                raise FileNotFoundError(
                    "Composite model " + str(i) + " does not exist!", path_model
                )
        # If heterogenous metalearner -> load metalearner model file
        if isinstance(self.ml_model, Metalearner_Base):
            path_model = os.path.join(directory_path, "metalearner.model.pickle")
            if not os.path.exists(path_model):
                raise FileNotFoundError("Metalearner model does not exist!", path_model)
            self.ml_model.load(path_model)

        # Update model directory
        self.cache_dir = directory_path


# -----------------------------------------------------#
#                     Subroutines                     #
# -----------------------------------------------------#
# Start, join, and safely retrieve a worker process' queued result.
#
# A worker's own try/except only catches regular Python exceptions. If the
# process instead dies from something that bypasses it (OOM-kill, a CUDA/driver
# segfault, ...), nothing is ever put on the queue and a bare `queue.get()`
# blocks forever with no diagnostic. This detects that case via the process'
# exitcode and raises immediately instead.
def __run_subprocess__(process, result_queue, label):
    process.start()
    # Drain the queue *before* joining the process. A child that puts a
    # payload larger than the OS pipe buffer (e.g. predictions for a large
    # ensemble/test split) blocks inside queue.put() until the parent reads
    # it -- joining first would wait on a child that can never exit, deadlocking
    # forever. Poll get() while the child is alive so a crash (OOM-kill, CUDA
    # fault) that never puts anything is still detected via exitcode.
    result = None
    got_result = False
    while True:
        try:
            result = result_queue.get(timeout=1)
            got_result = True
            break
        except Empty:
            if not process.is_alive():
                break
    process.join()
    if not got_result:
        exitcode = process.exitcode
        if exitcode is not None and exitcode < 0:
            try:
                sig_desc = signal.Signals(-exitcode).name
            except ValueError:
                sig_desc = str(-exitcode)
            raise RuntimeError(
                f"{label} subprocess (pid={process.pid}) was killed by signal "
                f"{sig_desc} (exitcode {exitcode}) before returning a result "
                "-- likely an OOM-kill or a native crash (e.g. CUDA/driver fault)."
            )
        raise RuntimeError(
            f"{label} subprocess (pid={process.pid}) exited with code {exitcode} "
            "without returning a result."
        )
    if isinstance(result, Exception):
        raise RuntimeError(f"{label} subprocess failed: {result}") from result
    return result


# Internal function for training a NeuralNetwork model in a separate process
def __training_process__(queue, data, model_paras, datagen_paras, train_paras):
    # Extract data
    train_x, train_y, train_m, test_x, test_y, test_m = data
    # Build training BatchLoader
    cv_train_gen = create_batch_loader(
        train_x,
        path_imagedir=datagen_paras["path_imagedir"],
        labels=train_y,
        metadata=train_m,
        batch_size=datagen_paras["batch_size"],
        data_aug=datagen_paras["data_aug"],
        seed=datagen_paras["seed"],
        subfunctions=datagen_paras["subfunctions"],
        shuffle=datagen_paras["shuffle"],
        standardize_mode=datagen_paras["standardize_mode"],
        resize=datagen_paras["resize"],
        grayscale=datagen_paras["grayscale"],
        two_dim=datagen_paras["two_dim"],
        prepare_images=datagen_paras["prepare_images"],
        sample_weights=datagen_paras["sample_weights"],
        image_format=datagen_paras["image_format"],
        loader=datagen_paras["loader"],
        num_workers=datagen_paras["num_workers"],
        **datagen_paras["kwargs"]
    )
    # Build validation BatchLoader
    cv_val_gen = create_batch_loader(
        test_x,
        path_imagedir=datagen_paras["path_imagedir"],
        labels=test_y,
        metadata=test_m,
        batch_size=datagen_paras["batch_size"],
        data_aug=None,
        seed=datagen_paras["seed"],
        subfunctions=datagen_paras["subfunctions"],
        shuffle=False,
        standardize_mode=datagen_paras["standardize_mode"],
        resize=datagen_paras["resize"],
        grayscale=datagen_paras["grayscale"],
        two_dim=datagen_paras["two_dim"],
        prepare_images=datagen_paras["prepare_images"],
        sample_weights=datagen_paras["sample_weights"],
        image_format=datagen_paras["image_format"],
        loader=datagen_paras["loader"],
        num_workers=datagen_paras["num_workers"],
        **datagen_paras["kwargs"]
    )
    # Create NeuralNetwork
    model = NeuralNetwork(**model_paras)
    # Start NeuralNetwork training
    try:
        cv_history = model.train(cv_train_gen, cv_val_gen, **train_paras)
        queue.put(cv_history)
    except Exception as exc:
        queue.put(exc)


# Internal function for inference with a fitted NeuralNetwork model in a separate process
def __prediction_process__(queue, model_paras, path_model, data_test, datagen_paras):
    # Extract data
    test_x, test_y, test_m = data_test
    # Create inference BatchLoader
    cv_pred_gen = create_batch_loader(
        test_x,
        path_imagedir=datagen_paras["path_imagedir"],
        labels=None,
        metadata=test_m,
        batch_size=datagen_paras["batch_size"],
        data_aug=None,
        seed=datagen_paras["seed"],
        subfunctions=datagen_paras["subfunctions"],
        shuffle=False,
        standardize_mode=datagen_paras["standardize_mode"],
        resize=datagen_paras["resize"],
        grayscale=datagen_paras["grayscale"],
        two_dim=datagen_paras["two_dim"],
        prepare_images=datagen_paras["prepare_images"],
        sample_weights=datagen_paras["sample_weights"],
        image_format=datagen_paras["image_format"],
        loader=datagen_paras["loader"],
        num_workers=datagen_paras["num_workers"],
        **datagen_paras["kwargs"]
    )
    # Create NeuralNetwork
    model = NeuralNetwork(**model_paras)
    # Load model weights from disk
    model.load(path_model)
    # Make prediction
    try:
        preds = model.predict(cv_pred_gen)
        queue.put(preds)
    except Exception as exc:
        queue.put(exc)
