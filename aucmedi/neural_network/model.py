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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
# Internal libraries/scripts
from aucmedi.neural_network.architectures import architecture_dict, \
                                                 supported_standardize_mode, \
                                                 Classifier

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
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
                                      resize=model.meta_input, standardize_mode=model.meta_standardize)
        model.train(datagen_train, epochs=50)
        # Do some predictions
        datagen_test = DataGenerator(samples[100:150], "images_dir/", labels=None,
                                     resize=model.meta_input, standardize_mode=model.meta_standardize)
        preds = model.predict(datagen_test)
        ```

    ??? example "Example: How to select an Architecture"
        ```python
        # 2D architecture
        my_model_a = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")
        # 3D architecture for multi-label classification (sigmoid activation)
        my_model_b = NeuralNetwork(n_labels=8, channels=3, architecture="3D.ResNet50",
                                    activation_output="sigmoid")
        # 2D architecture with custom input_shape
        my_model_c = NeuralNetwork(n_labels=8, channels=3, architecture="2D.Xception",
                                    input_shape=(512,512))
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
                              resize=my_model.meta_input,                  # (224,224)
                              standardize_mode=my_model.meta_standardize)  # "torch"
        ```

        ```python title="Manual way"
        from aucmedi.neural_network.architectures import Classifier, \
                                                         architecture_dict, \
                                                         supported_standardize_mode

        classification_head = Classifier(n_labels=4, activation_output="softmax")
        my_arch = architecture_dict["3D.DenseNet121"](classification_head,
                                                      channels=1,
                                                      input_shape=(128,128,128))

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
                                  meta_variables=10)

        my_dg = DataGenerator(samples, "images_dir/",
                              labels=None, metadata=my_metadata,
                              resize=my_model.meta_input,                  # (224,224)
                              standardize_mode=my_model.meta_standardize)  # "torch"
        ```
    """
    def __init__(self, n_labels, channels, input_shape=None, architecture=None,
                 pretrained_weights=False, loss="categorical_crossentropy",
                 metrics=["categorical_accuracy"], activation_output="softmax",
                 fcl_dropout=True, meta_variables=None, learning_rate=0.0001,
                 batch_queue_size=10, workers=1, multiprocessing=False,
                 verbose=1):
        """ Initialization function for creating a Neural Network (model) object.

        Args:
            n_labels (int):                         Number of classes/labels (important for the last layer).
            channels (int):                         Number of channels. Grayscale:1 or RGB:3.
            input_shape (tuple):                    Input shape of the batch imaging data (including channel axis).
                                                    If None is provided, the default input_shape for the architecture is selected
                                                    from the architecture dictionary.
            architecture (str or Architecture):     Key (str) or instance of a neural network model Architecture class instance.
                                                    If a string is provided, the corresponding architecture is selected from the architecture dictionary.
                                                    A string has to begin with either '3D.' or '2D' depending on the classification task.
                                                    By default, a 2D Vanilla Model is used as architecture.
            pretrained_weights (bool):              Option whether to utilize pretrained weights e.g. from ImageNet.
            loss (Metric Function):                 The metric function which is used as loss for training.
                                                    Any Metric Function defined in Keras, in aucmedi.neural_network.loss_functions or any custom
                                                    metric function, which follows the Keras metric guidelines, can be used.
            metrics (list of Metric Functions):     List of one or multiple Metric Functions, which will be shown during training.
                                                    Any Metric Function defined in Keras or any custom metric function, which follows the Keras
                                                    metric guidelines, can be used.
            activation_output (str):                Activation function which should be used in the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
                                                    Based on https://www.tensorflow.org/api_docs/python/tf/keras/activations.
            fcl_dropout (bool):                     Option whether to utilize an additional Dense & Dropout layer in the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            meta_variables (int):                   Number of metadata variables, which should be included in the classification head.
                                                    If `None`is provided, no metadata integration block will be added to the classification head
                                                    ([Classifier][aucmedi.neural_network.architectures.classifier]).
            learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
            batch_queue_size (int):                 The batch queue size is the number of previously prepared batches in the cache during runtime.
            workers (int):                          Number of workers/threads which preprocess batches during runtime.
            multiprocessing (bool):                 Option whether to utilize multi-processing for workers instead of threading .
            verbose (int):                          Option (0/1) how much information should be written to stdout.

        ???+ danger
            Class attributes can be modified also after initialization, at will.
            However, be aware of unexpected adverse effects (experimental)!

        Attributes:
            tf_epochs (int, default=5):             Transfer Learning configuration: Number of epochs with frozen layers except classification head.
            tf_lr_start (float, default=1e-4):      Transfer Learning configuration: Starting learning rate for frozen layer fitting.
            tf_lr_end (float, default=1e-5):        Transfer Learning configuration: Starting learning rate after layer unfreezing.
            meta_input (tuple of int):              Meta variable: Input shape of architecture which can be passed to a DataGenerator. For example: (224, 224).
            meta_standardize (str):                 Meta variable: Recommended standardize_mode of architecture which can be passed to a DataGenerator.
                                                    For example: "torch".
        """
        # Cache parameters
        self.n_labels = n_labels
        self.channels = channels
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.batch_queue_size = batch_queue_size
        self.workers = workers
        self.multiprocessing = multiprocessing
        self.pretrained_weights = pretrained_weights
        self.activation_output = activation_output
        self.fcl_dropout = fcl_dropout
        self.meta_variables = meta_variables
        self.verbose = verbose

        # Assemble architecture parameters
        arch_paras = {"channels":channels,
                      "pretrained_weights": pretrained_weights}
        if input_shape is not None : arch_paras["input_shape"] = input_shape
        # Assemble classifier parameters
        classifier_paras = {"n_labels": n_labels, "fcl_dropout": fcl_dropout,
                            "activation_output": activation_output}
        if meta_variables is not None:
            classifier_paras["meta_variables"] = meta_variables
        # Initialize classifier for the classification head
        arch_paras["classification_head"] = Classifier(**classifier_paras)
        # Initialize architecture if None provided
        if architecture is None:
            self.architecture = architecture_dict["2D.Vanilla"](**arch_paras)
            self.meta_standardize = "z-score"
        # Initialize passed architecture from aucmedi library
        elif isinstance(architecture, str) and architecture in architecture_dict:
            self.architecture = architecture_dict[architecture](**arch_paras)
            self.meta_standardize = supported_standardize_mode[architecture]
        # Initialize passed architecture as parameter
        else:
            self.architecture = architecture
            self.meta_standardize = None

        # Build model utilizing the selected architecture
        self.model = self.architecture.create_model()

        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=self.loss, metrics=self.metrics)

        # Obtain final input shape
        self.input_shape = self.architecture.input          # e.g. (224, 224, 3)
        self.meta_input = self.architecture.input[:-1]      # e.g. (224, 224) -> for DataGenerator
        # Cache starting weights
        self.initialization_weights = self.model.get_weights()

    #---------------------------------------------#
    #               Class Variables               #
    #---------------------------------------------#
    # Transfer Learning configurations
    tf_epochs = 5
    tf_lr_start = 1e-4
    tf_lr_end = 1e-5

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    # Training the Neural Network model
    def train(self, training_generator, validation_generator=None, epochs=20,
              iterations=None, callbacks=[], class_weights=None,
              transfer_learning=False):
        """ Fitting function for the Neural Network model performing a training process.

        It is also possible to pass custom Callback classes in order to obtain more information.

        If an optional validation [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator]
        is provided, a validation set is analyzed regularly during the training process (after each epoch).

        The transfer learning training runs two fitting processes.
        The first one with frozen base model layers and a high learning rate,
        whereas the second one with unfrozen layers and a small learning rate.

        ??? info "Keras History Objects for Transfer Learning"
            For the transfer learning training, two Keras history objects will be created.

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
            iterations (int):                       Number of iterations (batches) in a single epoch.
            callbacks (list of Callback classes):   A list of Callback classes for custom evaluation.
            class_weights (dictionary or list):     A list or dictionary of float values to handle class unbalance.
            transfer_learning (bool):               Option whether a transfer learning training should be performed. If true, a minimum of 5 epochs will be trained.

        Returns:
            history (dict):                   A history dictionary from a Keras history object which contains several logs.
        """
        # Running a standard training process
        if not transfer_learning:
            # Run training process with the Keras fit function
            history = self.model.fit(training_generator,
                                     validation_data=validation_generator,
                                     callbacks=callbacks, epochs=epochs,
                                     steps_per_epoch=iterations,
                                     class_weight=class_weights,
                                     workers=self.workers,
                                     use_multiprocessing=self.multiprocessing,
                                     max_queue_size=self.batch_queue_size,
                                     verbose=self.verbose)
            # Return logged history object
            return history.history

        # Running a transfer learning training process
        else:
            # Freeze all base model layers (all layers after "avg_pool")
            lever = False
            for layer in reversed(self.model.layers):
                if not lever and layer.name == "avg_pool" : lever = True
                elif lever : layer.trainable = False
            # Compile model with high learning rate
            self.model.compile(optimizer=Adam(learning_rate=self.tf_lr_start),
                               loss=self.loss, metrics=self.metrics)
            # Run first training with frozen layers
            history_start = self.model.fit(training_generator,
                                           validation_data=validation_generator,
                                           callbacks=callbacks,
                                           epochs=self.tf_epochs,
                                           steps_per_epoch=iterations,
                                           class_weight=class_weights,
                                           workers=self.workers,
                                           use_multiprocessing=self.multiprocessing,
                                           max_queue_size=self.batch_queue_size,
                                           verbose=self.verbose)
            # Unfreeze base model layers again
            for layer in self.model.layers:
                layer.trainable = True
            # Compile model with lower learning rate
            self.model.compile(optimizer=Adam(learning_rate=self.tf_lr_end),
                               loss=self.loss, metrics=self.metrics)
            # Reset data generators
            training_generator.reset()
            if validation_generator is not None : validation_generator.reset()
            # Run second training with unfrozed layers
            history_end = self.model.fit(training_generator,
                                         validation_data=validation_generator,
                                         callbacks=callbacks, epochs=epochs,
                                         initial_epoch=self.tf_epochs,
                                         steps_per_epoch=iterations,
                                         class_weight=class_weights,
                                         workers=self.workers,
                                         use_multiprocessing=self.multiprocessing,
                                         max_queue_size=self.batch_queue_size,
                                         verbose=self.verbose)
            # Combine logged history objects
            hs = {"tl_" + k: v for k, v in history_start.history.items()}       # prefix : tl for transfer learning
            he = {"ft_" + k: v for k, v in history_end.history.items()}         # prefix : ft for fine tuning
            history = {**hs, **he}
            # Return combined history objects
            return history

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    def predict(self, prediction_generator):
        """ Prediction function for the Neural Network model.

        The fitted model will predict classifications for the provided [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

        Args:
            prediction_generator (DataGenerator):   A data generator which will be used for inference.

        Returns:
            preds (numpy.ndarray):                  A NumPy array of predictions formatted with shape (n_samples, n_labels).
        """
        # Run inference process with the Keras predict function
        preds = self.model.predict(prediction_generator, workers=self.workers,
                                   max_queue_size=self.batch_queue_size,
                                   use_multiprocessing=self.multiprocessing,
                                   verbose=self.verbose)
        # Output predictions results
        return preds

    #---------------------------------------------#
    #               Model Management              #
    #---------------------------------------------#
    # Re-initialize model weights
    def reset_weights(self):
        """ Re-initialize weights of the neural network model.

        Useful for training multiple models with the same NeuralNetwork object.
        """
        self.model.set_weights(self.initialization_weights)

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
