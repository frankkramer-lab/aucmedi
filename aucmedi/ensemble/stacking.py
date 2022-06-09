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
from aucmedi.sampling import sampling_split
from aucmedi.ensemble.aggregate import aggregate_dict
from aucmedi.ensemble.metalearner import metalearner_dict
from aucmedi.ensemble.metalearner.ml_base import Metalearner_Base
from aucmedi.ensemble.aggregate.agg_base import Aggregate_Base

#-----------------------------------------------------#
#             Ensemble Learning: Stacking             #
#-----------------------------------------------------#
class Stacking:
    """ A Stacking class providing functionality for metalearner based ensemble learning.

    In contrast to single algorithm approaches, the ensemble of different deep convolutional neural network architectures
    (also called heterogeneous ensemble learning) showed strong benefits for overall performance in several studies.
    The idea of the Stacking technique is to utilize diverse and independent models by stacking another machine learning
    algorithm on top of these predictions.

    In AUCMEDI, a percentage split is applied on the dataset into the subsets: train, validation and ensemble.

    On the train and validation subsets, one or multiple predefined [NeuralNetwork][aucmedi.neural_network.model]
    models are trained, whereas on the ensemble subset a metalearner is fitted to merge
    [NeuralNetwork][aucmedi.neural_network.model] model predictions into a single one.

    ???+ example
        ```python
        # Initialize some NeuralNetwork models
        model_a = NeuralNetwork(n_labels=4, channels=3, architecture="2D.ResNet50")
        model_b = NeuralNetwork(n_labels=4, channels=3, architecture="2D.MobileNetV2")
        model_c = NeuralNetwork(n_labels=4, channels=3, architecture="2D.EfficientNetB1")

        # Initialize Stacking object
        el = Stacking(model_list=[model_a, model_b, model_c],
                      metalearner="logistic_regression")

        # Initialize training DataGenerator for complete training data
        datagen = DataGenerator(samples_train, "images_dir/",
                                labels=train_labels_ohe, batch_size=3,
                                resize=model_a.meta_input,
                                standardize_mode=model_a.meta_standardize)
        # Train neural network and metalearner models
        el.train(datagen, epochs=100)

        # Initialize testing DataGenerator for testing data
        test_gen = DataGenerator(samples_test, "images_dir/",
                                 resize=model_a.meta_input,
                                 standardize_mode=model_a.meta_standardize)
        # Run Inference
        preds = el.predict(test_gen)
        ```

    !!! warning "Training Time Increase"
        Stacking sequentially performs fitting processes for multiple models, which will drastically increase training time.

    ??? warning "DataGenerator re-initialization"
        The passed DataGenerator for the train() and predict() function of the Stacking class will be re-initialized!

        This can result in redundant image preparation if `prepare_images=True`.

    ??? info "Technical Details"
        For the training and inference process, each model will create an individual process via the Python multiprocessing package.

        This is crucial as TensorFlow does not fully support the VRAM memory garbage collection in GPUs,
        which is why more and more redundant data pile up with an increasing number of models.

        Via separate processes, it is possible to clean up the TensorFlow environment and rebuild it again for the next model.

    ??? reference "Reference for Ensemble Learning Techniques"
        Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2022).
        An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks.
        arXiv e-print: [https://arxiv.org/abs/2201.11440](https://arxiv.org/abs/2201.11440)
    """
    def __init__(self, model_list, metalearner="logistic_regression",
                 sampling=[0.7, 0.1, 0.2]):
        """ Initialization function for creating a Stacking object.

        Args:
            model_list (list of NeuralNetwork):        List of instances of AUCMEDI neural network class.
            metalearner (str, Metalearner or Aggregate):Metalearner class instance / a string for an AUCMEDI Metalearner,
                                                        or Aggregate function / a string for an AUCMEDI Aggregate function.
            sampling (list of float):                   List of percentage values with split sizes. Should be 3x percentage values
                                                        for heterogenous metalearner and 2x percentage values for homogeneous
                                                        Aggregate functions (must sum up to 1.0).
        """
        # Cache class variables
        self.model_list = model_list
        self.metalearner = metalearner
        self.sampling = sampling
        self.sampling_seed = 0
        self.cache_dir = None

        # Initialize Metalearner
        if isinstance(metalearner, str) and metalearner in metalearner_dict:
            self.ml_model = metalearner_dict[metalearner]()
        elif isinstance(metalearner, str) and metalearner in aggregate_dict:
            self.ml_model = aggregate_dict[metalearner]()
        elif isinstance(metalearner, Metalearner_Base) or \
             isinstance(metalearner, Aggregate_Base):
            self.ml_model = metalearner
        else : raise TypeError("Unknown type of Metalearner (neither known " + \
                               "ensembler nor Aggregate or Metalearner class)!")

        # Set multiprocessing method to spawn
        mp.set_start_method("spawn", force=True)

    def train(self, training_generator, epochs=20, iterations=None,
              callbacks=[], class_weights=None, transfer_learning=False,
              metalearner_fitting=True):
        """ Training function for fitting the provided Stacking models.

        The training data will be sampled according to a percentage split in which
        [DataGenerators][aucmedi.data_processing.data_generator.DataGenerator] for model training
        and validation as well as one for the metalearner training will be automatically created.

        It is also possible to pass custom Callback classes in order to obtain more information.

        For more information on the fitting process, check out [NeuralNetwork.train()][aucmedi.neural_network.model.NeuralNetwork.train].

        Args:
            training_generator (DataGenerator):     A data generator which will be used for training (will be split according
                                                    to percentage split sampling).
            epochs (int):                           Number of epochs. A single epoch is defined as one iteration through
                                                    the complete data set.
            iterations (int):                       Number of iterations (batches) in a single epoch.
            callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
            class_weights (dictionary or list):     A list or dictionary of float values to handle class unbalance.
            transfer_learning (bool):               Option whether a transfer learning training should be performed.
            metalearner_fitting (bool):             Option whether the Metalearner fitting process should be included in the
                                                    Stacking training process. The `train_metalearner()` function can also be
                                                    run manually (or repeatedly).
        Returns:
            history (dict):                   A history dictionary from a Keras history object which contains several logs.
        """
        temp_dg = training_generator    # Template DataGenerator variable for faster access
        history_stacking = {}           # Final history dictionary

        # Create temporary model directory
        self.cache_dir = tempfile.TemporaryDirectory(prefix="aucmedi.tmp.",
                                                     suffix=".stacking")

        # Obtain training data
        x = training_generator.samples
        y = training_generator.labels
        m = training_generator.metadata

        # Apply percentage split sampling
        ps_sampling = sampling_split(x, y, m, sampling=self.sampling,
                                     stratified=True, iterative=True,
                                     seed=self.sampling_seed)

        # Pack data according to sampling
        if len(ps_sampling[0]) == 3:
            data_train = ps_sampling[0]
            data_val = ps_sampling[1]
        else:
            data_train = (*ps_sampling[0], None)
            data_val = (*ps_sampling[1], None)

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

        # Sequentially iterate over model list
        for i in range(len(self.model_list)):
            # Extend Callback list
            path_model = os.path.join(self.cache_dir.name,
                                      "nn_" + str(i) + ".model.hdf5")
            cb_mc = ModelCheckpoint(path_model,
                                    monitor="val_loss", verbose=1,
                                    save_best_only=True, mode="min")
            cb_cl = CSVLogger(os.path.join(self.cache_dir.name,
                                                 "nn_" + str(i) + \
                                                 ".logs.csv"),
                              separator=',', append=True)
            callbacks.extend([cb_mc, cb_cl])

            # Start training process
            process_queue = mp.Queue()
            process_train = mp.Process(target=__training_process__,
                                       args=(process_queue,
                                             self.model_list[i],
                                             data_train,
                                             data_val,
                                             datagen_paras,
                                             parameters_training))
            process_train.start()
            process_train.join()
            nn_history = process_queue.get()
            # Combine logged history objects
            hnn = {"nn_" + str(i) + "." + k: v for k, v in nn_history.items()}
            history_stacking = {**history_stacking, **hnn}

        # Perform metalearner model training
        self.train_metalearner(temp_dg)

        # Return Stacking history object
        return history_stacking

    def train_metalearner(self, training_generator):
        """ Training function for fitting the Metalearner model.

        Function will be called automatically in the `train()` function if
        the parameter `metalearner_fitting` is true.

        However, this function can also be called multiple times for training
        different Metalearner types without the need of time-extensive
        re-training of the [NeuralNetwork][aucmedi.neural_network.model] models.

        Args:
            training_generator (DataGenerator):     A data generator which will be used for training (will be split according
                                                    to percentage split sampling).
        """
        # Skipping metalearner training if aggregate function
        if isinstance(self.ml_model, Aggregate_Base) : return

        temp_dg = training_generator    # Template DataGenerator variable for faster access
        preds_ensemble = []

        # Obtain training data
        x = training_generator.samples
        y = training_generator.labels
        m = training_generator.metadata

        # Apply percentage split sampling
        ps_sampling = sampling_split(x, y, m, sampling=self.sampling,
                                     stratified=True, iterative=True,
                                     seed=self.sampling_seed)

        # Pack data according to sampling
        if len(ps_sampling[0]) == 3 : data_ensemble = ps_sampling[2]
        else : data_ensemble = (*ps_sampling[2], None)

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

        # Identify path to model directory
        if isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            path_model_dir = self.cache_dir.name
        else : path_model_dir = self.cache_dir

        # Sequentially iterate over model list
        for i in range(len(self.model_list)):
            # Extend Callback list
            path_model = os.path.join(path_model_dir,
                                      "nn_" + str(i) + ".model.hdf5")

            # Start inference process for model i
            process_queue = mp.Queue()
            process_pred = mp.Process(target=__prediction_process__,
                                      args=(process_queue,
                                            self.model_list[i],
                                            path_model,
                                            data_ensemble,
                                            datagen_paras))
            process_pred.start()
            process_pred.join()
            preds = process_queue.get()

            # Append preds to ensemble
            preds_ensemble.append(preds)

        # Preprocess prediction ensemble
        preds_ensemble = np.array(preds_ensemble)
        preds_ensemble = np.swapaxes(preds_ensemble, 0, 1)
        s, m, c = preds_ensemble.shape
        x_stack = np.reshape(preds_ensemble, (s, m*c))

        # Start training of stacked metalearner
        if isinstance(self.ml_model, Metalearner_Base):
            (_, y_stack, _) = data_ensemble
            self.ml_model.train(x_stack, y_stack)
            # Store metalearner model to disk
            path_metalearner = os.path.join(path_model_dir,
                                            "metalearner.model.pickle")
            self.ml_model.dump(path_metalearner)

    def predict(self, prediction_generator, return_ensemble=False):
        """ Prediction function for Stacking.

        The fitted models and selected Metalearner will predict classifications for the provided
        [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        !!! info
            More about Metalearners can be found here: [Metelearner][aucmedi.ensemble.metalearner]

            More about Aggregate functions can be found here: [aggregate][aucmedi.ensemble.aggregate]

        Args:
            prediction_generator (DataGenerator):   A data generator which will be used for inference.
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
            raise FileNotFoundError("Stacking does not have a valid model cache directory!")

        # Initialize some variables
        temp_dg = prediction_generator
        preds_ensemble = []
        preds_final = []

        # Extract data
        data_test = (temp_dg.samples, temp_dg.labels, temp_dg.metadata)

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

        # Identify path to model directory
        if isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            path_model_dir = self.cache_dir.name
        else : path_model_dir = self.cache_dir

        # Sequentially iterate over model list
        for i in range(len(self.model_list)):
            path_model = os.path.join(path_model_dir,
                                      "nn_" + str(i) + ".model.hdf5")

            # Start inference process for model i
            process_queue = mp.Queue()
            process_pred = mp.Process(target=__prediction_process__,
                                      args=(process_queue,
                                            self.model_list[i],
                                            path_model,
                                            data_test,
                                            datagen_paras))
            process_pred.start()
            process_pred.join()
            preds = process_queue.get()

            # Append preds to ensemble
            preds_ensemble.append(preds)

        # Preprocess prediction ensemble
        preds_ensemble = np.array(preds_ensemble)
        preds_ensemble = np.swapaxes(preds_ensemble, 0, 1)

        # Apply heterogenous metalearner
        if isinstance(self.ml_model, Metalearner_Base):
            s, m, c = preds_ensemble.shape
            x_stack = np.reshape(preds_ensemble, (s, m*c))
            preds_final = self.ml_model.predict(data=x_stack)
        # Apply homogeneous aggregate function
        elif isinstance(self.ml_model, Aggregate_Base):
            for i in range(preds_ensemble.shape[0]):
                pred_sample = self.ml_model.aggregate(preds_ensemble[i,:,:])
                preds_final.append(pred_sample)

        # Convert prediction list to NumPy
        preds_final = np.asarray(preds_final)

        # Return ensembled predictions
        if return_ensemble : return preds_final, np.swapaxes(preds_ensemble,1,0)
        else : return preds_final

    # Dump model to file
    def dump(self, directory_path):
        """ Store temporary Stacking models directory permanently to disk at desired location.

        If the model directory is a provided path which is already persistent on the disk,
        the directory is copied in order to keep original data persistent.

        Args:
            directory_path (str):       Path to store the model directory on disk.
        """
        if self.cache_dir is None:
            raise FileNotFoundError("Stacking does not have a valid model cache directory!")
        elif isinstance(self.cache_dir, tempfile.TemporaryDirectory):
            shutil.copytree(self.cache_dir.name, directory_path)
            self.cache_dir.cleanup()
            self.cache_dir = directory_path
        else:
            shutil.copytree(self.cache_dir, directory_path)
            self.cache_dir = directory_path

    # Load model from file
    def load(self, directory_path):
        """ Load a Stacking model directory which can be used for Metalearner based inference.

        Args:
            directory_path (str):       Input path, from which the Stacking models will be loaded.
        """
        # Check directory existence
        if not os.path.exists(directory_path):
            raise FileNotFoundError("Provided model directory path does not exist!",
                                    directory_path)
        # Check model existence
        for i in range(len(self.model_list)):
            path_model = os.path.join(directory_path,
                                      "nn_" + str(i) + ".model.hdf5")
            if not os.path.exists(path_model):
                raise FileNotFoundError("Stacking model " + str(i) + \
                                        " does not exist!", path_model)
        # If heterogenous metalearner -> load metalearner model file
        if isinstance(self.ml_model, Metalearner_Base):
            path_model = os.path.join(directory_path,
                                      "metalearner.model.pickle")
            if not os.path.exists(path_model):
                raise FileNotFoundError("Metalearner model does not exist!",
                                        path_model)
            self.ml_model.load(path_model)

        # Update model directory
        self.cache_dir = directory_path

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Internal function for training a NeuralNetwork model in a separate process
def __training_process__(queue, model, data_train, data_val, datagen_paras,
                         train_paras):
    # Extract data
    (train_x, train_y, train_m) = data_train
    (val_x, val_y, val_m) = data_val
    # Build training DataGenerator
    nn_train_gen = DataGenerator(train_x,
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
    nn_val_gen = DataGenerator(val_x,
                               path_imagedir=datagen_paras["path_imagedir"],
                               labels=val_y,
                               metadata=val_m,
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
    nn_history = model.train(nn_train_gen, nn_val_gen, **train_paras)
    # Store result in cache (which will be returned by the process queue)
    queue.put(nn_history)

# Internal function for inference with a fitted NeuralNetwork model in a separate process
def __prediction_process__(queue, model, path_model, data_test, datagen_paras):
    # Extract data
    (test_x, test_y, test_m) = data_test
    # Create inference DataGenerator
    nn_pred_gen = DataGenerator(test_x,
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
                                prepare_images=datagen_paras["prepare_images"],
                                sample_weights=datagen_paras["sample_weights"],
                                image_format=datagen_paras["image_format"],
                                loader=datagen_paras["loader"],
                                workers=datagen_paras["workers"],
                                **datagen_paras["kwargs"])
    # Load model weights from disk
    model.load(path_model)
    # Make prediction
    preds = model.predict(nn_pred_gen)
    # Store prediction results in cache (which will be returned by the process queue)
    queue.put(preds)
