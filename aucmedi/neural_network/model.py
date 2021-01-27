#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
# Internal libraries/scripts
from aucmedi.neural_network.architectures import architecture_dict, Vanilla

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network
class Neural_Network:
    """ Initialization function for creating a Neural Network (model) object.
    This class provides functionality for handling all model methods.

    With an initialized Neural Network model instance, it is possible to run training and predictions.

    Args:

        n_labels (Integer):                     Number of classes/labels (important for the last layer).
        channels (Integer):                     Number of channels. Grayscale:1 or RGB:3.
        input_shape (Tuple):                    Input shape of the batch imaging data (including channel axis).
        architecture (Architecture):            Instance of a neural network model Architecture class instance.
                                                By default, a Vanilla Model is used as Architecture.
        pretrained_weights (Boolean):           Option whether to utilize pretrained weights e.g. for ImageNet.
        loss (Metric Function):                 The metric function which is used as loss for training.
                                                Any Metric Function defined in Keras, in aucmedi.neural_network.loss_functions or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        metrics (List of Metric Functions):     List of one or multiple Metric Functions, which will be shown during training.
                                                Any Metric Function defined in Keras or any custom metric function, which follows the Keras
                                                metric guidelines, can be used.
        out_activation (String):                Activation function which should be used in the last classification layer.
        dropout (Boolean):                      Option whether to utilize a dropout layer in the last classification layer.
        learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
        batch_queue_size (integer):             The batch queue size is the number of previously prepared batches in the cache during runtime.
        Number of workers (integer):            Number of workers/threads which preprocess batches during runtime.
        verbose (integer):                      Option (0/1) how much information should be written to stdout.
    """
    def __init__(self, n_labels, channels, input_shape=None, architecture=None,
                 pretrained_weights=False, loss="categorical_crossentropy",
                 metrics=["categorical_accuracy"], activation_output="softmax",
                 dropout=True, learninig_rate=0.001, batch_queue_size=10,
                 workers=1, verbose=1):
        # Cache parameters
        self.n_labels = n_labels
        self.channels = channels
        self.loss = loss
        self.metrics = metrics
        self.learninig_rate = learninig_rate
        self.batch_queue_size = batch_queue_size
        self.workers = workers
        self.pretrained_weights = pretrained_weights
        self.activation_output = activation_output
        self.dropout = dropout
        self.verbose = verbose

        # Assemble architecture parameters
        arch_paras = {"channels":channels}
        if input_shape is not None : arch_paras["input_shape"] = input_shape
        # Initialize architecture if None provided
        if architecture is None:
            self.architecture = Vanilla(**arch_paras)
        # Initialize passed architecture from aucmedi library
        elif isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
        # Initialize passed architecture as parameter
        else : self.architecture = architecture

        # Build model utilizing the selected architecture
        model_paras = {"n_labels": n_labels, "dropout": dropout,
                       "out_activation": activation_output,
                       "pretrained_weights": pretrained_weights}
        self.model = self.architecture.create_model(**model_paras)

        # Compile model
        self.model.compile(optimizer=Adam(lr=learninig_rate),
                           loss=self.loss, metrics=self.metrics)

        # Obtain final input shape
        self.input_shape = self.architecture.input
        # Cache starting weights
        self.initialization_weights = self.model.get_weights()

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    """ Fitting function for the Neural Network model performing a training process.
        It is also possible to pass custom Callback classes in order to obtain more information.

        If an optional validation generator is provided, a validation set is analyzed regularly
        during the training process (after each epoch).

    Args:
        training_generator (DataGenerator):     A data generator which will be used for training.
        validation_generator (DataGenerator):   A data generator which will be used for validation.
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through
                                                the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
        class_weights (dictionary or list):     A list or dictionary of float values to handle class unbalance.

    Returns:
        A Keras history object (dictionary) which contains several logs.
    """
    # Training the Neural Network model
    def train(self, training_generator, validation_generator=None, epochs=20,
              iterations=None, callbacks=[], class_weights=None):
        # Run training process with the Keras fit function
        history = self.model.fit(training_generator,
                                 validation_data=validation_generator,
                                 callbacks=callbacks, epochs=epochs,
                                 steps_per_epoch=iterations,
                                 class_weight=class_weights,
                                 workers=self.workers,
                                 max_queue_size=self.batch_queue_size,
                                 verbose=self.verbose)
        # Return logged history object
        return history

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    """ Prediction function for the Neural Network model. The fitted model will predict classifications
        for the provided data generator.

    Args:
        prediction_generator (DataGenerator):   A data generator which will be used for training.
    """
    def predict(self, prediction_generator):
        # Run inference process with the Keras predict function
        preds = self.model.predict(prediction_generator, workers=self.workers,
                                   max_queue_size=self.batch_queue_size,
                                   verbose=self.verbose)
        # Output predictions results
        return preds

    #---------------------------------------------#
    #               Model Management              #
    #---------------------------------------------#
    # Re-initialize model weights
    def reset_weights(self):
        self.model.set_weights(self.initialization_weights)

    # Dump model to file
    def dump(self, file_path):
        self.model.save(file_path)

    # Load model from file
    def load(self, file_path, custom_objects={}):
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.learninig_rate),
                           loss=self.loss, metrics=self.metrics)
