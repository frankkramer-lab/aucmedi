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
from copy import deepcopy
import tempfile
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import multiprocessing as mp
import numpy as np
# Internal libraries
from aucmedi import DataGenerator
from aucmedi.sampling import sampling_kfold
from aucmedi.ensemble.aggregate import aggregate_dict

#-----------------------------------------------------#
#              Ensemble Learning: Bagging             #
#-----------------------------------------------------#
class Bagging:
    """ A Bagging class providing functionality for cross-validation based ensemble learning.

    """
    def __init__(self, model, k_fold=3):
        """ Initialization function for creating a Bagging object.
        """
        # Cache class variables
        self.model_template = model
        self.k_fold = k_fold
        self.cache_dir = None

        # Set multiprocessing method to spawn
        mp.set_start_method("spawn")


    def train(self, training_generator, epochs=20, iterations=None,
              callbacks=[], class_weights=None, transfer_learning=False):
        """ asd

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

    def predict(self, prediction_generator, aggregate="mean"):
        """ asd
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
        return preds_final

    # Dump model to file
    def dump(self, file_path):
        """ Store model to disk.

        Recommended to utilize the file format ".hdf5".

        Args:
            file_path (str):    Path to store the model on disk.
        """
        self.model.save(file_path)


    # Load model from file
    def load(self, file_path, custom_objects={}):
        """ Load neural network model and its weights from a file.

        After loading, the model will be compiled.

        If loading a model in ".hdf5" format, it is not necessary to define any custom_objects.

        Args:
            file_path (str):            Input path, from which the model will be loaded.
            custom_objects (dict):      Dictionary of custom objects for compiling
                                        (e.g. non-TensorFlow based loss functions or architectures).
        """
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss=self.loss, metrics=self.metrics)

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Internal function for training a Neural_Network model in a separate process
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
    # Start Neural_Network training
    cv_history = model.train(cv_train_gen, cv_val_gen, **train_paras)
    # Store result in cache (which will be returned by the process queue)
    queue.put(cv_history)

# Internal function for inference with a fitted Neural_Network model in a separate process
def __prediction_process__(queue, model, path_model, datagen_paras):
    # Create inference DataGenerator
    cv_pred_gen = DataGenerator(datagen_paras["samples"],
                                path_imagedir=datagen_paras["path_imagedir"],
                                labels=None,
                                metadata=datagen_paras["metadata"],
                                batch_size=datagen_paras["batch_size"],
                                data_aug=datagen_paras["metadata"],
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
    preds = model.predict(cv_pred_gen)
    # Store prediction results in cache (which will be returned by the process queue)
    queue.put(preds)
