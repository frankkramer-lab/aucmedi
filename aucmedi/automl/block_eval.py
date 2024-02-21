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
import pandas as pd
import numpy as np
import re
# Internal libraries
from aucmedi import *
from aucmedi.evaluation import evaluate_performance

#-----------------------------------------------------#
#            Building Blocks for Evaluation           #
#-----------------------------------------------------#
def block_evaluate(config):
    """ Internal code block for AutoML evaluation.

    This function is called by the Command-Line-Interface (CLI) of AUCMEDI.

    Args:
        config (dict):                      Configuration dictionary containing all required
                                            parameters for performing an AutoML evaluation.

    The following attributes are stored in the `config` dictionary:

    Attributes:
        path_imagedir (str):                Path to the directory containing the ground truth images.
        path_gt (str):                      Path to the index/class annotation file if required. (only for 'csv' interface).
        path_pred (str):                    Path to the input file in which predicted csv file is stored.
        path_evaldir (str):                 Path to the directory in which evaluation figures and tables should be stored.
        ohe (bool):                         Boolean option whether annotation data is sparse categorical or one-hot encoded.
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
    if not os.path.exists(config["path_evaldir"]):
        os.mkdir(config["path_evaldir"])

    # Read prediction csv
    df_pred = pd.read_csv(config["path_pred"])

    # Create ground truth pandas dataframe
    df_index = pd.DataFrame(data={"SAMPLE": index_list})
    df_gt_data = pd.DataFrame(data=class_ohe, columns=class_names)
    df_gt = pd.concat([df_index, df_gt_data], axis=1, sort=False)


    # Verify - maybe there is a file path encoded in the index?
    if os.path.sep in df_gt.iloc[0,0]:
        samples_split = df_gt["SAMPLE"].str.split(pat=os.path.sep,
                                                  expand=False)
        df_gt["SAMPLE"] = samples_split.str[-1]
    # Verify - maybe the image format is present in the index?
    if image_format is None and bool(re.fullmatch(r"^.*\.[A-Za-z]+$",
                                                  df_gt.iloc[0,0])):
        samples_split = df_gt["SAMPLE"].str.split(pat=".",
                                                  expand=False)
        df_gt["SAMPLE"] = samples_split.str[:-1].str.join(".")

    # Merge dataframes to verify correct order
    df_merged = df_pred.merge(df_gt, on="SAMPLE", suffixes=("_pd", "_gt"))

    # Extract pd and gt again to NumPy
    data_pd = df_merged.iloc[:, 1:(class_n+1)].to_numpy()
    data_gt = df_merged.iloc[:, (class_n+1):].to_numpy()

    # Identify task (multi-class vs multi-label)
    if np.sum(data_pd) > (class_ohe.shape[0] + 1.5) : multi_label = True
    else : multi_label = False

    # Evaluate performance via AUCMEDI evaluation submodule
    evaluate_performance(data_pd, data_gt,
                         out_path=config["path_evaldir"],
                         class_names=class_names,
                         multi_label=multi_label,
                         metrics_threshold=0.5,
                         suffix=None,
                         store_csv=True,
                         plot_barplot=True,
                         plot_confusion_matrix=True,
                         plot_roc_curve=True)
