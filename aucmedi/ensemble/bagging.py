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
from aucmedi import NeuralNetwork, create_data_loader
from aucmedi.data_processing.wrapper_loader import WrapperLoader
from aucmedi.sampling import sampling_kfold
from aucmedi.ensemble.aggregate import aggregate_dict


# -----------------------------------------------------#
#              Generator Resolution Helper            #
# -----------------------------------------------------#
# Resolves the two generator wrapper styles Bagging accepts -- a plain torch
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
#              Ensemble Learning: Bagging             #
# -----------------------------------------------------#
class Bagging:
    """A Bagging class providing functionality for cross-validation based ensemble learning.

    Homogeneous model ensembles can be defined as multiple models consisting of the same algorithm, hyperparameters,
    or architecture. The Bagging technique is based on improved training dataset sampling and a popular homogeneous
    ensemble learning technique. In contrast to a standard single training/validation split, which results in a single
    model, Bagging consists of training multiple models on randomly drawn subsets from the dataset.

    In AUCMEDI, a k-fold cross-validation is applied on the dataset resulting in k models.

    ???+ example
        ```python
        # Initialize NeuralNetwork model
        model = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet50")

        # Initialize Bagging object for 3-fold cross-validation
        el = Bagging(model, k_fold=3)


        # Initialize training BatchLoader for complete training data
        datagen = create_data_loader(samples_train, "images_dir/",
                                labels=train_labels_ohe, batch_size=3,
                                resize=model.arch_resolution,
                                standardize_mode=model.arch_standardize)
        # Train models
        el.train(datagen, epochs=100)


        # Initialize testing BatchLoader for testing data
        test_gen = create_data_loader(samples_test, "images_dir/",
                                 resize=model.arch_resolution,
                                 standardize_mode=model.arch_standardize)
        # Run Inference with majority vote aggregation
        preds = el.predict(test_gen, aggregate="majority_vote")
        ```

    !!! warning "Training Time Increase"
        Bagging sequentially performs fitting processes for multiple models (commonly `k_fold=3` up to `k_fold=10`),
        which will drastically increase training time.

    ??? warning "BatchLoader re-initialization"
        The passed BatchLoader for the train() and predict() function of the Bagging class will be re-initialized!

        This can result in redundant image preparation if `prepare_images=True`.

    ??? warning "NeuralNetwork re-initialization"
        The passed NeuralNetwork for the train() and predict() function of the Bagging class will be re-initialized!

        Attention: Metrics are not passed to the processes due to pickling issues.

    ??? info "Technical Details"
        For the training and inference process, each model will create an individual process via the Python multiprocessing package.

    ??? reference "Reference for Ensemble Learning Techniques"
        Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2022).
        An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks.
        arXiv e-print: [https://arxiv.org/abs/2201.11440](https://arxiv.org/abs/2201.11440)
    """

    def __init__(self, model, k_fold=3):
        """Initialization function for creating a Bagging object.

        Args:
            model (NeuralNetwork):         Instance of an AUCMEDI neural network class.
            k_fold (int):                   Number of folds (k) for the Cross-Validation. Must be at least 2.
        """
        # Cache class variables
        self.model_template = model
        self.k_fold = k_fold
        self.cache_dir = None

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
    ):
        """Training function for the Bagging models which performs a k-fold cross-validation model fitting.

        The training data will be sampled according to a k-fold cross-validation in which a validation
        WrapperLoader will be automatically created.

        It is also possible to pass custom Callback classes in order to obtain more information.

        For more information on the fitting process, check out [NeuralNetwork.train()][aucmedi.neural_network.model.NeuralNetwork.train].

        Args:
            training_generator (WrapperLoader or DataLoader):     A generator which will be used for training (will be split according to k-fold sampling).
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

        Returns:
            history (dict):                   A history dictionary which contains several logs.
        """
        # Resolve generator type (DataLoader or WrapperLoader)
        training_generator, self.num_workers = __resolve_template_generator__(
            training_generator
        )

        history_bagging = {}  # Final history dictionary

        # Create temporary model directory
        self.cache_dir = tempfile.TemporaryDirectory(
            prefix="aucmedi.tmp.", suffix=".bagging"
        )

        # Obtain training data
        x = training_generator.samples
        y = training_generator.labels
        m = training_generator.metadata

        # Apply cross-validaton sampling
        cv_sampling = sampling_kfold(
            x, y, m, n_splits=self.k_fold, stratified=True, iterative=True
        )

        # Sequentially iterate over all folds
        for i, fold in enumerate(cv_sampling):
            # Pack data into a tuple
            if len(fold) == 4:
                train_x, train_y, test_x, test_y = fold
                data = (train_x, train_y, None, test_x, test_y, None)
            else:
                data = fold

            # Create model specific callback list
            callbacks_model = callbacks.copy()
            # Extend Callback list
            cb_mc = ModelCheckpoint(
                os.path.join(self.cache_dir.name, "cv_" + str(i) + ".model.pt"),
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
                "n_labels": self.model_template.n_labels,
                "channels": self.model_template.channels,
                "input_resolution": self.model_template.arch_resolution,
                "architecture": self.model_template.architecture,
                "pretrained_weights": self.model_template.pretrained_weights,
                "loss": self.model_template.loss,
                "metrics": None,
                "activation_output": self.model_template.activation_output,
                "fcl_dropout": self.model_template.fcl_dropout,
                "n_meta_variables": self.model_template.n_meta_variables,
            }

            # Gather DataGenerator parameters
            datagen_paras = {
                "path_imagedir": training_generator.path_imagedir,
                "batch_size": training_generator.batch_size,
                "data_aug": training_generator.data_aug,
                "seed": training_generator.seed,
                "subfunctions": training_generator.subfunctions,
                "shuffle": training_generator.shuffle,
                "standardize_mode": training_generator.standardize_mode,
                "resize": training_generator.resize,
                "grayscale": training_generator.grayscale,
                "prepare_images": training_generator.prepare_images,
                "sample_weights": training_generator.sample_weights,
                "image_format": training_generator.image_format,
                "loader": training_generator.sample_loader,
                "num_workers": self.num_workers,
                "kwargs": training_generator.kwargs,
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
                    model_paras,
                    data,
                    datagen_paras,
                    parameters_training,
                ),
            )
            cv_history = __run_subprocess__(
                process_train, process_queue, label=f"Training (fold {i})"
            )
            # Combine logged history objects
            hcv = {"cv_" + str(i) + "." + k: v for k, v in cv_history.items()}
            history_bagging = {**history_bagging, **hcv}

        # Return Bagging history object
        return history_bagging

    def predict(self, prediction_generator, aggregate="mean", return_ensemble=False):
        """Prediction function for the Bagging models.

        The fitted models will predict classifications for the provided [WrapperLoader][aucmedi.data_processing.wrapper_loader.WrapperLoader].

        The inclusion of the Aggregate function can be achieved in multiple ways:

        - self-initialization with an AUCMEDI Aggregate function,
        - use a string key to call an AUCMEDI Aggregate function by name, or
        - implementing a custom Aggregate function by extending the [AUCMEDI base class for Aggregate functions][aucmedi.ensemble.aggregate.agg_base]

        !!! info
            Description and list of implemented Aggregate functions can be found here:
            [Aggregate][aucmedi.ensemble.aggregate]

        Args:
            prediction_generator (WrapperLoader or DataLoader):   A generator which will be used for inference.
                                                                Must be a WrapperLoader or a torch DataLoader wrapping a DataGenerator.
            aggregate (str or aggregate Function):  Aggregate function class instance or a string for an AUCMEDI Aggregate function.
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
                "Bagging does not have a valid model cache directory!"
            )

        # Initialize aggregate function if required
        if isinstance(aggregate, str) and aggregate in aggregate_dict:
            agg_fun = aggregate_dict[aggregate]()
        else:
            agg_fun = aggregate

        # Resolve generator type (DataLoader or WrapperLoader)
        prediction_generator, self.num_workers = __resolve_template_generator__(
            prediction_generator
        )

        # Initialize some variables
        preds_ensemble = []
        preds_final = []

        # Gather DataGenerator parameters
        datagen_paras = {
            "samples": prediction_generator.samples,
            "metadata": prediction_generator.metadata,
            "path_imagedir": prediction_generator.path_imagedir,
            "batch_size": prediction_generator.batch_size,
            "data_aug": prediction_generator.data_aug,
            "seed": prediction_generator.seed,
            "subfunctions": prediction_generator.subfunctions,
            "shuffle": prediction_generator.shuffle,
            "standardize_mode": prediction_generator.standardize_mode,
            "resize": prediction_generator.resize,
            "grayscale": prediction_generator.grayscale,
            "prepare_images": prediction_generator.prepare_images,
            "sample_weights": prediction_generator.sample_weights,
            "image_format": prediction_generator.image_format,
            "loader": prediction_generator.sample_loader,
            "num_workers": self.num_workers,
            "kwargs": prediction_generator.kwargs,
        }

        # Identify path to model directory
        if isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            path_model_dir = self.cache_dir.name
        else:
            path_model_dir = self.cache_dir

        # Sequentially iterate over all fold models
        for i in range(self.k_fold):
            # Identify path to fitted model
            path_model = os.path.join(path_model_dir, "cv_" + str(i) + ".model.pt")

            # Gather NeuralNetwork parameters
            model_paras = {
                "n_labels": self.model_template.n_labels,
                "channels": self.model_template.channels,
                "input_resolution": self.model_template.arch_resolution,
                "architecture": self.model_template.architecture,
                "pretrained_weights": self.model_template.pretrained_weights,
                "loss": self.model_template.loss,
                "metrics": None,
                "activation_output": self.model_template.activation_output,
                "fcl_dropout": self.model_template.fcl_dropout,
                "n_meta_variables": self.model_template.n_meta_variables,
            }

            # Start inference process for fold i
            process_queue = mp.Queue()
            process_pred = mp.Process(
                target=__prediction_process__,
                args=(process_queue, model_paras, path_model, datagen_paras),
            )
            preds = __run_subprocess__(
                process_pred, process_queue, label=f"Prediction (fold {i})"
            )

            # Append to prediction ensemble
            preds_ensemble.append(preds)

        # Aggregate predictions
        preds_ensemble = np.array(preds_ensemble)
        for i in range(0, len(prediction_generator.samples)):
            pred_sample = agg_fun.aggregate(preds_ensemble[:, i, :])
            preds_final.append(pred_sample)

        # Convert prediction list to NumPy
        preds_final = np.asarray(preds_final)

        # Return ensembled predictions
        if return_ensemble:
            return preds_final, preds_ensemble
        else:
            return preds_final

    # Dump model to file
    def dump(self, directory_path):
        """Store temporary Bagging model directory permanently to disk at desired location.

        If the model directory is a provided path which is already persistent on the disk,
        the directory is copied in order to keep original data persistent.

        Args:
            directory_path (str):       Path to store the model directory on disk.
        """
        if self.cache_dir is None:
            raise FileNotFoundError(
                "Bagging does not have a valid model cache directory!"
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
        """Load a Bagging model directory which can be used for aggregated inference.

        Args:
            directory_path (str):       Input path, from which the Bagging models will be loaded.
        """
        # Check directory existence
        if not os.path.exists(directory_path):
            raise FileNotFoundError(
                "Provided model directory path does not exist!", directory_path
            )
        # Check model existence
        for i in range(self.k_fold):
            path_model = os.path.join(directory_path, "cv_" + str(i) + ".model.pt")
            if not os.path.exists(path_model):
                raise FileNotFoundError(
                    "Bagging model for fold " + str(i) + " does not exist!", path_model
                )
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
def __training_process__(queue, model_paras, data, datagen_paras, train_paras):
    train_x, train_y, train_m, test_x, test_y, test_m = data
    # Build training BatchLoader
    cv_train_gen = create_data_loader(
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
        prepare_images=datagen_paras["prepare_images"],
        sample_weights=datagen_paras["sample_weights"],
        image_format=datagen_paras["image_format"],
        loader=datagen_paras["loader"],
        num_workers=datagen_paras["num_workers"],
        **datagen_paras["kwargs"]
    )
    # Build validation BatchLoader
    cv_val_gen = create_data_loader(
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
def __prediction_process__(queue, model_paras, path_model, datagen_paras):
    # Create inference BatchLoader
    cv_pred_gen = create_data_loader(
        datagen_paras["samples"],
        path_imagedir=datagen_paras["path_imagedir"],
        labels=None,
        metadata=datagen_paras["metadata"],
        batch_size=datagen_paras["batch_size"],
        data_aug=datagen_paras["data_aug"],
        seed=datagen_paras["seed"],
        subfunctions=datagen_paras["subfunctions"],
        shuffle=datagen_paras["shuffle"],
        standardize_mode=datagen_paras["standardize_mode"],
        resize=datagen_paras["resize"],
        grayscale=datagen_paras["grayscale"],
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
