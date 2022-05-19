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
        self.model_list = []
        self.cache_dir = None

        # Create k models based on template
        for i in range(k_fold):
            model_clone = deepcopy(model)
            self.model_list.append(model_clone)

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
            # Access current fold data
            if len(fold) == 4:
                (train_x, train_y, test_x, test_y) = fold
                train_m = None
                test_m = None
            else : (train_x, train_y, train_m, test_x, test_y, test_m) = fold

            # Build training DataGenerator
            cv_train_gen = DataGenerator(train_x,
                                         path_imagedir=temp_dg.path_imagedir,
                                         labels=train_y,
                                         metadata=train_m,
                                         batch_size=temp_dg.batch_size,
                                         data_aug=temp_dg.data_aug,
                                         seed=temp_dg.seed,
                                         subfunctions=temp_dg.subfunctions,
                                         shuffle=temp_dg.shuffle,
                                         standardize_mode=temp_dg.standardize_mode,
                                         resize=temp_dg.resize,
                                         grayscale=temp_dg.grayscale,
                                         prepare_images=temp_dg.prepare_images,
                                         sample_weights=temp_dg.sample_weights,
                                         image_format=temp_dg.image_format,
                                         loader=temp_dg.sample_loader,
                                         workers=temp_dg.workers,
                                         **temp_dg.kwargs)
            # Build validation DataGenerator
            cv_val_gen = DataGenerator(test_x,
                                       path_imagedir=temp_dg.path_imagedir,
                                       labels=test_y,
                                       metadata=test_m,
                                       batch_size=temp_dg.batch_size,
                                       data_aug=None,
                                       seed=temp_dg.seed,
                                       subfunctions=temp_dg.subfunctions,
                                       shuffle=False,
                                       standardize_mode=temp_dg.standardize_mode,
                                       resize=temp_dg.resize,
                                       grayscale=temp_dg.grayscale,
                                       prepare_images=temp_dg.prepare_images,
                                       sample_weights=temp_dg.sample_weights,
                                       image_format=temp_dg.image_format,
                                       loader=temp_dg.sample_loader,
                                       workers=temp_dg.workers,
                                       **temp_dg.kwargs)

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

            # Perform training process
            cv_history = self.model_list[i].train(cv_train_gen,
                                                  cv_val_gen,
                                                  epochs=epochs,
                                                  iterations=iterations,
                                                  callbacks=callbacks,
                                                  class_weights=class_weights,
                                                  transfer_learning=transfer_learning)
            # Combine logged history objects
            hcv = {"cv_" + str(i) + "." + k: v for k, v in cv_history.items()}
            history_bagging = {**history_bagging, **hcv}

        # Return Bagging history object
        return history_bagging


    def predict(self, prediction_generator, aggregate=""):
        pass
        # for loop
            # model.predict
        # aggregate
        # return


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
