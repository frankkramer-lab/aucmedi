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
import json
import pandas as pd
# Internal libraries
from aucmedi import *
from aucmedi.data_processing.io_loader import image_loader, sitk_loader
from aucmedi.data_processing.subfunctions import *
from aucmedi.ensemble import *
from aucmedi.xai import xai_decoder

#-----------------------------------------------------#
#            Building Blocks for Inference            #
#-----------------------------------------------------#
def block_predict(config):
    """ Internal code block for AutoML inference.

    This function is called by the Command-Line-Interface (CLI) of AUCMEDI.

    Args:
        config (dict):                      Configuration dictionary containing all required
                                            parameters for performing an AutoML inference.

    The following attributes are stored in the `config` dictionary:

    Attributes:
        path_imagedir (str):                Path to the directory containing the images for prediction.
        path_modeldir (str):                Path to the model directory in which fitted model weights and metadata are stored.
        path_pred (str):                    Path to the output file in which predicted csv file should be stored.
        xai_method (str or None):           Key for XAI method.
        xai_directory (str or None):        Path to the output directory in which predicted image xai heatmaps should be stored.
        batch_size (int):                   Number of samples inside a single batch.
        workers (int):                      Number of workers/threads which preprocess batches during runtime.
    """
    # Peak into the dataset via the input interface
    ds = input_interface("directory",
                         config["path_imagedir"],
                         path_data=None,
                         training=False,
                         ohe=False,
                         image_format=None)
    (index_list, _, _, _, image_format) = ds

    # Verify existence of input directory
    if not os.path.exists(config["path_modeldir"]):
        raise FileNotFoundError(config["path_modeldir"])

    # Load metadata from training
    path_meta = os.path.join(config["path_modeldir"], "meta.training.json")
    with open(path_meta, "r") as json_file:
        meta_training = json.load(json_file)

    # Define neural network parameters
    nn_paras = {"n_labels": 1,                                  # placeholder
                "channels": 1,                                  # placeholder
    }
    # Select input shape for 3D
    if meta_training["three_dim"]:
        nn_paras["input_shape"] = tuple(meta_training["shape_3D"])

    # Subfunctions
    sf_list = []
    if meta_training["three_dim"]:
        sf_norm = Standardize(mode="grayscale")
        sf_pad = Padding(mode="constant", shape=meta_training["shape_3D"])
        sf_crop = Crop(shape=meta_training["shape_3D"], mode="random")
        sf_chromer = Chromer(target="rgb")
        sf_list.extend([sf_norm, sf_pad, sf_crop, sf_chromer])

    # Define parameters for DataGenerator
    paras_datagen = {
        "path_imagedir": config["path_imagedir"],
        "batch_size": config["batch_size"],
        "img_aug": None,
        "subfunctions": sf_list,
        "prepare_images": False,
        "sample_weights": None,
        "seed": None,
        "image_format": image_format,
        "workers": config["workers"],
        "shuffle": False,
        "grayscale": False,
    }
    if not meta_training["three_dim"] : paras_datagen["loader"] = image_loader
    else : paras_datagen["loader"] = sitk_loader

    # Apply MIC pipelines
    if meta_training["analysis"] == "minimal":
        # Setup neural network
        if not meta_training["three_dim"]:
            arch_dim = "2D." + meta_training["architecture"]
        else : arch_dim = "3D." + meta_training["architecture"]
        model = NeuralNetwork(architecture=arch_dim, **nn_paras)

        # Build DataGenerator
        pred_gen = DataGenerator(samples=index_list,
                                 labels=None,
                                 resize=model.meta_input,
                                 standardize_mode=model.meta_standardize,
                                 **paras_datagen)
        # Load model
        path_model = os.path.join(config["path_modeldir"], "model.last.keras")
        model.load(path_model)
        # Start model inference
        preds = model.predict(prediction_generator=pred_gen)
    elif meta_training["analysis"] == "standard":
        # Setup neural network
        if not meta_training["three_dim"]:
            arch_dim = "2D." + meta_training["architecture"]
        else : arch_dim = "3D." + meta_training["architecture"]
        model = NeuralNetwork(architecture=arch_dim, **nn_paras)

        # Build DataGenerator
        pred_gen = DataGenerator(samples=index_list,
                                 labels=None,
                                 resize=model.meta_input,
                                 standardize_mode=model.meta_standardize,
                                 **paras_datagen)
        # Load model
        path_model = os.path.join(config["path_modeldir"],
                                  "model.best_loss.keras")
        model.load(path_model)
        # Start model inference via Augmenting
        preds = predict_augmenting(model, pred_gen)
    else:
        # Build multi-model list
        model_list = []
        for arch in meta_training["architecture"]:
            if not meta_training["three_dim"] : arch_dim = "2D." + arch
            else : arch_dim = "3D." + arch
            model_part = NeuralNetwork(architecture=arch_dim, **nn_paras)
            model_list.append(model_part)
        el = Composite(model_list, metalearner=meta_training["metalearner"],
                       k_fold=len(meta_training["architecture"]))

        # Build DataGenerator
        pred_gen = DataGenerator(samples=index_list,
                                 labels=None,
                                 resize=None,
                                 standardize_mode=None,
                                 **paras_datagen)
        # Load composite model directory
        el.load(config["path_modeldir"])
        # Start model inference via ensemble learning
        preds = el.predict(pred_gen)

    # Create prediction dataset
    df_index = pd.DataFrame(data={"SAMPLE": index_list})
    df_pd = pd.DataFrame(data=preds, columns=meta_training["class_names"])
    df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
    df_merged.sort_values(by=["SAMPLE"], inplace=True)
    # Store predictions to disk
    df_merged.to_csv(config["path_pred"], index=False)

    # Create XAI heatmaps
    if config["xai_method"] is not None and config["xai_directory"] is not None:
        if meta_training["analysis"] == "advanced":
            raise ValueError("XAI is only supported for single model pipelines!")
        # Create xai output directory
        if not os.path.exists(config["xai_directory"]):
            os.mkdir(config["xai_directory"])
        # Run XAI decoder
        xai_decoder(pred_gen, model, preds=preds, method=config["xai_method"],
                    layerName=None, alpha=0.4, out_path=config["xai_directory"])
