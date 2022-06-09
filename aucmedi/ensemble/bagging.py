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
import os
import tempfile
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import multiprocessing as mp
import numpy as np
import shutil
# Internal libraries
from aucmedi import DataGenerator
from aucmedi.sampling import sampling_kfold
from aucmedi.ensemble.aggregate import aggregate_dict

#-----------------------------------------------------#
#              Ensemble Learning: Bagging             #
#-----------------------------------------------------#
class Bagging:
    """ A Bagging class providing functionality for cross-validation based ensemble learning.

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


        # Initialize training DataGenerator for complete training data
        datagen = DataGenerator(samples_train, "images_dir/",
                                labels=train_labels_ohe, batch_size=3,
                                resize=model.meta_input,
                                standardize_mode=model.meta_standardize)
        # Train models
        el.train(datagen, epochs=100)


        # Initialize testing DataGenerator for testing data
        test_gen = DataGenerator(samples_test, "images_dir/",
                                 resize=model.meta_input,
                                 standardize_mode=model.meta_standardize)
        # Run Inference with majority vote aggregation
        preds = el.predict(test_gen, aggregate="majority_vote")
        ```

    !!! warning "Training Time Increase"
        Bagging sequentially performs fitting processes for multiple models (commonly `k_fold=3` up to `k_fold=10`),
        which will drastically increase training time.

    ??? warning "DataGenerator re-initialization"
        The passed DataGenerator for the train() and predict() function of the Bagging class will be re-initialized!

        This can result in redundant image preparation if `prepare_images=True`.

    ??? info "Technical Details"
        For the training and inference process, each model will create an individual process via the Python multiprocessing package.

        This is crucial as TensorFlow does not fully support the VRAM memory garbage collection in GPUs,
        which is why more and more redundant data pile up with an increasing number of k-fold.

        Via separate processes, it is possible to clean up the TensorFlow environment and rebuild it again for the next fold model.

    ??? reference "Reference for Ensemble Learning Techniques"
        Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2022).
        An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks.
        arXiv e-print: [https://arxiv.org/abs/2201.11440](https://arxiv.org/abs/2201.11440)
    """
    def __init__(self, model, k_fold=3):
        """ Initialization function for creating a Bagging object.

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

    def train(self, training_generator, epochs=20, iterations=None,
              callbacks=[], class_weights=None, transfer_learning=False):
        """ Training function for the Bagging models which performs a k-fold cross-validation model fitting.

        The training data will be sampled according to a k-fold cross-validation in which a validation
        [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator] will be automatically created.

        It is also possible to pass custom Callback classes in order to obtain more information.

        For more information on the fitting process, check out [NeuralNetwork.train()][aucmedi.neural_network.model.NeuralNetwork.train].

        Args:
            training_generator (DataGenerator):     A data generator which will be used for training (will be split according to k-fold sampling).
            epochs (int):                           Number of epochs. A single epoch is defined as one iteration through
                                                    the complete data set.
            iterations (int):                       Number of iterations (batches) in a single epoch.
            callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
            class_weights (dictionary or list):     A list or dictionary of float values to handle class imbalance.
            transfer_learning (bool):               Option whether a transfer learning training should be performed.

        Returns:
            history (dict):                   A history dictionary from a Keras history object which contains several logs.
        """
        temp_dg = training_generator    # Template DataGenerator variable for faster access
        history_bagging = {}            # Final history dictionary

        # Create temporary model directory
        self.cache_dir = tempfile.TemporaryDirectory(prefix="aucmedi.tmp.",
                                                     suffix=".bagging")

        # Obtain training data
        x = training_generator.samples
        y = training_generator.labels
        m = training_generator.metadata

        # Apply cross-validaton sampling
        cv_sampling = sampling_kfold(x, y, m, n_splits=self.k_fold,
                                     stratified=True, iterative=True)

        # Sequentially iterate over all folds
        for i, fold in enumerate(cv_sampling):
            # Pack data into a tuple
            if len(fold) == 4:
                (train_x, train_y, test_x, test_y) = fold
                data = (train_x, train_y, None, test_x, test_y, None)
            else : data = fold

            # Extend Callback list
            cb_mc = ModelCheckpoint(os.path.join(self.cache_dir.name,
                                                 "cv_" + str(i) + \
                                                 ".model.hdf5"),
                                    monitor="val_loss", verbose=1,
                                    save_best_only=True, mode="min")
            cb_cl = CSVLogger(os.path.join(self.cache_dir.name,
                                                 "cv_" + str(i) + \
                                                 ".logs.csv"),
                              separator=',', append=True)
            callbacks.extend([cb_mc, cb_cl])

            # Gather DataGenerator parameters
            datagen_paras = {"path_imagedir": temp_dg.path_imagedir,
                             "batch_size": temp_dg.batch_size,
                             "data_aug": temp_dg.data_aug,
                             "seed": temp_dg.seed,
                             "subfunctions": temp_dg.subfunctions,
                             "shuffle": temp_dg.shuffle,
                             "standardize_mode": temp_dg.standardize_mode,
                             "resize": temp_dg.resize,
                             "grayscale": temp_dg.grayscale,
                             "prepare_images": temp_dg.prepare_images,
                             "sample_weights": temp_dg.sample_weights,
                             "image_format": temp_dg.image_format,
                             "loader": temp_dg.sample_loader,
                             "workers": temp_dg.workers,
                             "kwargs": temp_dg.kwargs
            }

            # Gather training parameters
            parameters_training = {"epochs": epochs,
                                   "iterations": iterations,
                                   "callbacks": callbacks,
                                   "class_weights": class_weights,
                                   "transfer_learning": transfer_learning
            }

            # Start training process
            process_queue = mp.Queue()
            process_train = mp.Process(target=__training_process__,
                                       args=(process_queue,
                                             self.model_template,
                                             data,
                                             datagen_paras,
                                             parameters_training))
            process_train.start()
            process_train.join()
            cv_history = process_queue.get()
            # Combine logged history objects
            hcv = {"cv_" + str(i) + "." + k: v for k, v in cv_history.items()}
            history_bagging = {**history_bagging, **hcv}

        # Return Bagging history object
        return history_bagging

    def predict(self, prediction_generator, aggregate="mean",
                return_ensemble=False):
        """ Prediction function for the Bagging models.

        The fitted models will predict classifications for the provided [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        The inclusion of the Aggregate function can be achieved in multiple ways:

        - self-initialization with an AUCMEDI Aggregate function,
        - use a string key to call an AUCMEDI Aggregate function by name, or
        - implementing a custom Aggregate function by extending the [AUCMEDI base class for Aggregate functions][aucmedi.ensemble.aggregate.agg_base]

        !!! info
            Description and list of implemented Aggregate functions can be found here:
            [Aggregate][aucmedi.ensemble.aggregate]

        Args:
            prediction_generator (DataGenerator):   A data generator which will be used for inference.
            aggregate (str or aggregate Function):  Aggregate function class instance or a string for an AUCMEDI Aggregate function.
            return_ensemble (bool):                 Option, whether gathered ensemble of predictions should be returned.

        Returns:
            preds (numpy.ndarray):                  A NumPy array of predictions formatted with shape (n_samples, n_labels).
            ensemble (numpy.ndarray):               Optional ensemble of predictions: Will be only passed if `return_ensemble=True`.
                                                    Shape (n_models, n_samples, n_labels).
        """
        # Verify if there is a linked cache dictionary
        con_tmp = (isinstance(self.cache_dir, tempfile.TemporaryDirectory) and \
                   os.path.exists(self.cache_dir.name))
        con_var = (self.cache_dir is not None and \
                   not isinstance(self.cache_dir, tempfile.TemporaryDirectory) \
                   and os.path.exists(self.cache_dir))
        if not con_tmp and not con_var:
            raise FileNotFoundError("Bagging does not have a valid model cache directory!")

        # Initialize aggregate function if required
        if isinstance(aggregate, str) and aggregate in aggregate_dict:
            agg_fun = aggregate_dict[aggregate]()
        else : agg_fun = aggregate

        # Initialize some variables
        temp_dg = prediction_generator
        preds_ensemble = []
        preds_final = []

        # Gather DataGenerator parameters
        datagen_paras = {"samples": temp_dg.samples,
                         "metadata": temp_dg.metadata,
                         "path_imagedir": temp_dg.path_imagedir,
                         "batch_size": temp_dg.batch_size,
                         "data_aug": temp_dg.data_aug,
                         "seed": temp_dg.seed,
                         "subfunctions": temp_dg.subfunctions,
                         "shuffle": temp_dg.shuffle,
                         "standardize_mode": temp_dg.standardize_mode,
                         "resize": temp_dg.resize,
                         "grayscale": temp_dg.grayscale,
                         "prepare_images": temp_dg.prepare_images,
                         "sample_weights": temp_dg.sample_weights,
                         "image_format": temp_dg.image_format,
                         "loader": temp_dg.sample_loader,
                         "workers": temp_dg.workers,
                         "kwargs": temp_dg.kwargs
        }

        # Sequentially iterate over all fold models
        for i in range(self.k_fold):
            # Identify path to fitted model
            if isinstance(self.cache_dir, tempfile.TemporaryDirectory):
                path_model_dir = self.cache_dir.name
            else : path_model_dir = self.cache_dir
            path_model = os.path.join(path_model_dir,
                                      "cv_" + str(i) + ".model.hdf5")

            # Start inference process for fold i
            process_queue = mp.Queue()
            process_pred = mp.Process(target=__prediction_process__,
                                      args=(process_queue,
                                            self.model_template,
                                            path_model,
                                            datagen_paras))
            process_pred.start()
            process_pred.join()
            preds = process_queue.get()

            # Append to prediction ensemble
            preds_ensemble.append(preds)

        # Aggregate predictions
        preds_ensemble = np.array(preds_ensemble)
        for i in range(0, len(temp_dg.samples)):
            pred_sample = agg_fun.aggregate(preds_ensemble[:,i,:])
            preds_final.append(pred_sample)

        # Convert prediction list to NumPy
        preds_final = np.asarray(preds_final)
        
        # Return ensembled predictions
        if return_ensemble : return preds_final, preds_ensemble
        else : return preds_final

    # Dump model to file
    def dump(self, directory_path):
        """ Store temporary Bagging model directory permanently to disk at desired location.

        If the model directory is a provided path which is already persistent on the disk,
        the directory is copied in order to keep original data persistent.

        Args:
            directory_path (str):       Path to store the model directory on disk.
        """
        if self.cache_dir is None:
            raise FileNotFoundError("Bagging does not have a valid model cache directory!")
        elif isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            shutil.copytree(self.cache_dir.name, directory_path)
            self.cache_dir.cleanup()
            self.cache_dir = directory_path
        else:
            shutil.copytree(self.cache_dir, directory_path)
            self.cache_dir = directory_path

    # Load model from file
    def load(self, directory_path):
        """ Load a Bagging model directory which can be used for aggregated inference.

        Args:
            directory_path (str):       Input path, from which the Bagging models will be loaded.
        """
        # Check directory existence
        if not os.path.exists(directory_path):
            raise FileNotFoundError("Provided model directory path does not exist!",
                                    directory_path)
        # Check model existence
        for i in range(self.k_fold):
            path_model = os.path.join(directory_path,
                                      "cv_" + str(i) + ".model.hdf5")
            if not os.path.exists(path_model):
                raise FileNotFoundError("Bagging model for fold " + str(i) + \
                                        " does not exist!", path_model)
        # Update model directory
        self.cache_dir = directory_path

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Internal function for training a NeuralNetwork model in a separate process
def __training_process__(queue, model, data, datagen_paras, train_paras):
    (train_x, train_y, train_m, test_x, test_y, test_m) = data
    # Build training DataGenerator
    cv_train_gen = DataGenerator(train_x,
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
                                 workers=datagen_paras["workers"],
                                 **datagen_paras["kwargs"])
    # Build validation DataGenerator
    cv_val_gen = DataGenerator(test_x,
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
                               workers=datagen_paras["workers"],
                               **datagen_paras["kwargs"])
    # Start NeuralNetwork training
    cv_history = model.train(cv_train_gen, cv_val_gen, **train_paras)
    # Store result in cache (which will be returned by the process queue)
    queue.put(cv_history)

# Internal function for inference with a fitted NeuralNetwork model in a separate process
def __prediction_process__(queue, model, path_model, datagen_paras):
    # Create inference DataGenerator
    cv_pred_gen = DataGenerator(datagen_paras["samples"],
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
                                workers=datagen_paras["workers"],
                                **datagen_paras["kwargs"])
    # Load model weights from disk
    model.load(path_model)
    # Make prediction
    preds = model.predict(cv_pred_gen)
    # Store prediction results in cache (which will be returned by the process queue)
    queue.put(preds)
