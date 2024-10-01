#==============================================================================#
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
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import numpy as np
import json
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
# Internal libraries
from aucmedi import *
from aucmedi.data_processing.io_loader import image_loader, sitk_loader
from aucmedi.sampling import sampling_split
from aucmedi.utils.class_weights import *
from aucmedi.data_processing.subfunctions import *
from aucmedi.neural_network.loss_functions import *
from aucmedi.ensemble import *
from aucmedi.evaluation import evaluate_fitting

#-----------------------------------------------------#
#            Building Blocks for Training             #
#-----------------------------------------------------#
def block_train(config):
    """ Internal code block for AutoML training.

    This function is called by the Command-Line-Interface (CLI) of AUCMEDI.

    Args:
        config (dict):                      Configuration dictionary containing all required
                                            parameters for performing an AutoML training.

    The following attributes are stored in the `config` dictionary:

    Attributes:
        path_imagedir (str):                Path to the directory containing the images.
        path_modeldir (str):                Path to the output directory in which fitted models and metadata are stored.
        path_gt (str):                      Path to the index/class annotation file if required. (only for 'csv' interface).
        analysis (str):                     Analysis mode for the AutoML training. Options: `["minimal", "standard", "advanced"]`.
        ohe (bool):                         Boolean option whether annotation data is sparse categorical or one-hot encoded.
        three_dim (bool):                   Boolean, whether data is 2D or 3D.
        shape_3D (tuple of int):            Desired input shape of 3D volume for architecture (will be cropped).
        epochs (int):                       Number of epochs. A single epoch is defined as one iteration through
                                            the complete data set.
        batch_size (int):                   Number of samples inside a single batch.
        workers (int):                      Number of workers/threads which preprocess batches during runtime.
        metalearner (str):                  Key for Metalearner or Aggregate function.
        architecture (str or list of str):  Key (str) of a neural network model Architecture class instance.
    """
    # Obtain interface
    if config["path_gt"] is None : config["interface"] = "directory"
    else : config["interface"] = "csv"
    # Peak into the dataset via the input interface
    ds = input_interface(config["interface"],
                         config["path_imagedir"],
                         path_data=config["path_gt"],
                         training=True,
                         ohe=config["ohe"],
                         image_format=None)
    (index_list, class_ohe, class_n, class_names, image_format) = ds

    # Create output directory
    if not os.path.exists(config["path_modeldir"]):
        os.mkdir(config["path_modeldir"])

    # Identify task (multi-class vs multi-label)
    if np.sum(class_ohe) > class_ohe.shape[0] : config["multi_label"] = True
    else : config["multi_label"] = False

    # Sanity check on multi-label metalearner
    multilabel_metalearner_supported = ["mlp", "k_neighbors", "random_forest",
                                        "weighted_mean", "best_model",
                                        "decision_tree", "mean", "median"]
    if config["multi_label"] and config["analysis"] == "advanced" and \
       config["metalearner"] not in multilabel_metalearner_supported:
        raise ValueError("Non-compatible metalearner selected for multi-label"\
                         + " classification. Supported metalearner:",
                          multilabel_metalearner_supported)

    # Store meta information
    config["class_names"] = class_names
    path_meta = os.path.join(config["path_modeldir"], "meta.training.json")
    with open(path_meta, "w") as json_io:
        json.dump(config, json_io)

    # Define Callbacks
    callbacks = []
    if config["analysis"] == "standard":
        cb_loss = ModelCheckpoint(os.path.join(config["path_modeldir"],
                                               "model.best_loss.keras"),
                                  monitor="val_loss", verbose=1,
                                  save_best_only=True)
        callbacks.append(cb_loss)
    if config["analysis"] in ["minimal", "standard"]:
        cb_cl = CSVLogger(os.path.join(config["path_modeldir"],
                                       "logs.training.csv"),
                          separator=',', append=True)
        callbacks.append(cb_cl)
    if config["analysis"] != "minimal":
        cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='min', min_lr=1e-7)
        cb_es = EarlyStopping(monitor='val_loss', patience=12, verbose=1)
        callbacks.extend([cb_lr, cb_es])

    # Initialize loss function for multi-class
    if not config["multi_label"]:
        # Compute class weights
        class_weights, _ = compute_class_weights(ohe_array=class_ohe)
        # Initialize focal loss
        loss = categorical_focal_loss(class_weights)
    # Initialize loss function for multi-label
    else:
        # Compute class weights
        class_weights = compute_multilabel_weights(ohe_array=class_ohe)
        # Initialize focal loss
        loss = multilabel_focal_loss(class_weights)

    # Define neural network parameters
    nn_paras = {"n_labels": class_n,
                "channels": 3,
                "loss": loss,
                "metrics": [AUC(100)],
                "pretrained_weights": True,
    }
    # Select input shape for 3D
    if config["three_dim"] : nn_paras["input_shape"] = config["shape_3D"]
    # Select task type
    if config["multi_label"] : nn_paras["activation_output"] = "sigmoid"
    else : nn_paras["activation_output"] = "softmax"

    # Initialize Augmentation for 2D image data
    if not config["three_dim"]:
        data_aug = ImageAugmentation(flip=True, rotate=True, scale=False,
                                     brightness=True, contrast=True,
                                     saturation=False, hue=False, crop=False,
                                     grid_distortion=False, compression=False,
                                     gamma=True, gaussian_noise=False,
                                     gaussian_blur=False, downscaling=False,
                                     elastic_transform=True)
    # Initialize Augmentation for 3D volume data
    elif config["three_dim"]:
        data_aug = BatchgeneratorsAugmentation(image_shape=config["shape_3D"],
                        mirror=True, rotate=True, scale=True,
                        elastic_transform=True, gaussian_noise=False,
                        brightness=False, contrast=False, gamma=True)
    else : data_aug = None

    # Subfunctions
    sf_list = []
    if config["three_dim"]:
        sf_norm = Standardize(mode="grayscale")
        sf_pad = Padding(mode="constant", shape=config["shape_3D"])
        sf_crop = Crop(shape=config["shape_3D"], mode="random")
        sf_chromer = Chromer(target="rgb")
        sf_list.extend([sf_norm, sf_pad, sf_crop, sf_chromer])

    # Define parameters for DataGenerator
    paras_datagen = {
        "path_imagedir": config["path_imagedir"],
        "batch_size": config["batch_size"],
        "img_aug": data_aug,
        "subfunctions": sf_list,
        "prepare_images": False,
        "sample_weights": None,
        "seed": None,
        "image_format": image_format,
        "workers": config["workers"],
    }
    if not config["three_dim"] : paras_datagen["loader"] = image_loader
    else : paras_datagen["loader"] = sitk_loader

    # Gather training parameters
    paras_train = {
        "epochs": config["epochs"],
        "iterations": None,
        "callbacks": callbacks,
        "class_weights": None,
        "transfer_learning": True,
    }

    # Apply MIC pipelines
    if config["analysis"] == "minimal":
        # Setup neural network
        if not config["three_dim"] : arch_dim = "2D." + config["architecture"]
        else : arch_dim = "3D." + config["architecture"]
        model = NeuralNetwork(architecture=arch_dim, **nn_paras)

        # Build DataGenerator
        train_gen = DataGenerator(samples=index_list,
                                  labels=class_ohe,
                                  shuffle=True,
                                  resize=model.meta_input,
                                  standardize_mode=model.meta_standardize,
                                  **paras_datagen)

        # Start model training
        hist = model.train(training_generator=train_gen, **paras_train)
        # Store model
        path_model = os.path.join(config["path_modeldir"], "model.last.keras")
        model.dump(path_model)
    elif config["analysis"] == "standard":
        # Setup neural network
        if not config["three_dim"] : arch_dim = "2D." + config["architecture"]
        else : arch_dim = "3D." + config["architecture"]
        model = NeuralNetwork(architecture=arch_dim, **nn_paras)

        # Apply percentage split sampling
        ps_sampling = sampling_split(index_list, class_ohe,
                                     sampling=[0.85, 0.15],
                                     stratified=True, iterative=True,
                                     seed=0)

        # Build DataGenerator
        train_gen = DataGenerator(samples=ps_sampling[0][0],
                                  labels=ps_sampling[0][1],
                                  shuffle=True,
                                  resize=model.meta_input,
                                  standardize_mode=model.meta_standardize,
                                  **paras_datagen)
        val_gen = DataGenerator(samples=ps_sampling[1][0],
                                labels=ps_sampling[1][1],
                                shuffle=False,
                                resize=model.meta_input,
                                standardize_mode=model.meta_standardize,
                                **paras_datagen)

        # Start model training
        hist = model.train(training_generator=train_gen,
                           validation_generator=val_gen,
                           **paras_train)
        # Store model
        path_model = os.path.join(config["path_modeldir"], "model.last.keras")
        model.dump(path_model)
    else:
        # Sanity check of architecutre config
        if not isinstance(config["architecture"], list):
            raise ValueError("key 'architecture' in config has to be a list " \
                             + "if 'advanced' was selected as analysis.")
        # Build multi-model list
        model_list = []
        for arch in config["architecture"]:
            if not config["three_dim"] : arch_dim = "2D." + arch
            else : arch_dim = "3D." + arch
            model_part = NeuralNetwork(architecture=arch_dim, **nn_paras)
            model_list.append(model_part)
        el = Composite(model_list, metalearner=config["metalearner"],
                       k_fold=len(config["architecture"]))

        # Build DataGenerator
        train_gen = DataGenerator(samples=index_list,
                                  labels=class_ohe,
                                  shuffle=True,
                                  resize=None,
                                  standardize_mode=None,
                                  **paras_datagen)
        # Start model training
        hist = el.train(training_generator=train_gen, **paras_train)
        # Store model directory
        el.dump(config["path_modeldir"])

    # Plot fitting history
    evaluate_fitting(train_history=hist, out_path=config["path_modeldir"])
