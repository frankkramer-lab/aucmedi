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
# External Libraries
import numpy as np
import pandas as pd
import os
from plotnine import *
# Internal libraries/scripts
from aucmedi.evaluation.metrics import *

#-----------------------------------------------------#
#           Evaluation - Compare Performance          #
#-----------------------------------------------------#
def evaluate_comparison(pred_list,
                        labels,
                        out_path,
                        model_names=None,
                        class_names=None,
                        multi_label=False,
                        metrics_threshold=0.5,
                        macro_average_classes=False,
                        suffix=None):
    """ Function for performance comparison evaluation based on predictions from multiple models.

    ???+ example
        ```python
        # Import libraries
        from aucmedi import *
        from aucmedi.evaluation import *
        from aucmedi.ensemble import *

        # Load data
        ds = input_interface(interface="csv",                       # Interface type
                             path_imagedir="dataset/images/",
                             path_data="dataset/annotations.csv",
                             ohe=False, col_sample="ID", col_class="diagnosis")
        (samples, class_ohe, nclasses, class_names, image_format) = ds

        # Initialize model
        model_a = NeuralNetwork(n_labels=8, channels=3, architecture="2D.ResNet50")

        # Initialize Bagging object for 3-fold cross-validation
        el = Bagging(model_a, k_fold=3)

        # Do some predictions via Bagging (return also all prediction ensembles)
        datagen_test = DataGenerator(samples, "dataset/images/", labels=None,
                                     resize=model.meta_input, standardize_mode=model.meta_standardize)
        pred_merged, pred_ensemble = el.predict(datagen_test, return_ensemble=True)

        # Pass prediction ensemble to evaluation function
        evaluate_comparison(pred_ensemble, class_ohe, out_path="./", class_names=class_names)


        # Do some predictions with manually initialized models
        model_b = NeuralNetwork(n_labels=8, channels=3, architecture="2D.DenseNet121")
        model_c = NeuralNetwork(n_labels=8, channels=3, architecture="2D.MobileNetV2")

        pred_a = model_a.predict(datagen_test)
        pred_b = model_b.predict(datagen_test)
        pred_c = model_c.predict(datagen_test)

        pred_ensemble = [pred_a, pred_b, pred_c]

        # Pass prediction ensemble to evaluation function
        evaluate_comparison(pred_ensemble, class_ohe, out_path="./", class_names=class_names)
        ```

    Created files in directory of `out_path`:

    - "plot.comparison.beside.png"
    - "plot.comparison.gain.png"

    ???+ info "Preview for Bar Plot"
        ![Evaluation_Comparison_Beside](../../images/evaluation.plot.comparison.beside.png)

        Predictions based on [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) with
        macro-averaged class-wise metrics.

    ???+ info "Preview for Confusion Matrix"
        ![Evaluation_Comparison_Gain](../../images/evaluation.plot.comparison.gain.png)

        Predictions based on [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/).

    Args:
        pred_list (list of numpy.ndarray):  A list of NumPy arrays containing predictions from multiple models formatted with shape
                                            (n_models, n_samples, n_labels). Provided by [NeuralNetwork][aucmedi.neural_network.model].
        labels (numpy.ndarray):             Classification list with One-Hot Encoding. Provided by
                                            [input_interface][aucmedi.data_processing.io_data.input_interface].
        out_path (str):                     Path to directory in which plotted figures are stored.
        model_names (list of str):          List of names for corresponding models which are for visualization. If not provided (`None`
                                            provided), model index of `pred_list` will be used.
        class_names (list of str):          List of names for corresponding classes. Used for evaluation. Provided by
                                            [input_interface][aucmedi.data_processing.io_data.input_interface].
                                            If not provided (`None` provided), class indices will be used.
        multi_label (bool):                 Option, whether task is multi-label based (has impact on evaluation).
        metrics_threshold (float):          Only required if 'multi_label==True`. Threshold value if prediction is positive.
                                            Used in metric computation for CSV and bar plot.
        macro_average_classes (bool):       Option, whether classes should be macro-averaged in order to increase visualization overview.
        suffix (str):                       Special suffix to add in the created figure filename.

    Returns:
        df_merged (pandas.DataFrame):       Dataframe containing the merged metrics of all models.
        df_gain (pandas.DataFrame):         Dataframe containing performance gain compared to first model.
    """
    # Identify number of labels
    n_labels = labels.shape[-1]
    # Identify prediction threshold
    if multi_label : threshold = metrics_threshold
    else : threshold = None

    # Compute metric dataframe for each mode
    df_list = []
    for m in range(0, len(pred_list)):
        metrics = compute_metrics(pred_list[m], labels, n_labels, threshold)

        # Rename class association in metrics dataframe
        class_mapping = {}
        if class_names is not None:
            for c in range(len(class_names)):
                class_mapping[c] = class_names[c]
            metrics["class"].replace(class_mapping, inplace=True)
        if class_names is None:
            metrics["class"] = pd.Categorical(metrics["class"])

        # Assign model name to dataframe
        if model_names is not None : metrics["model"] = model_names[m]
        else : metrics["model"] = "model_" + str(m)

        # Optional: Macro average classes
        if macro_average_classes:
            metrics_avg = metrics.groupby(["metric", "model"])[["score"]].mean()
            metrics = metrics_avg.reset_index()

        # Append to dataframe list
        df_list.append(metrics)
    # Combine dataframes
    df_merged = pd.concat(df_list, axis=0, ignore_index=True)

    # Generate comparison beside plot
    evalby_beside(df_merged, out_path, suffix)

    # Generate comparison gain plot
    df_gain = evalby_gain(df_merged, out_path, suffix)

    # Return combined and gain dataframe
    return df_merged, df_gain

#-----------------------------------------------------#
#           Evaluation Comparison - Beside            #
#-----------------------------------------------------#
def evalby_beside(df, out_path, suffix=None):
    # Remove confusion matrix from dataframe
    df = df[~df["metric"].isin(["TN", "FN", "FP", "TP"])]

    # Plot metric results class-wise
    if "class" in df.columns:
        fig = (ggplot(df, aes("model", "score", fill="model"))
                  + geom_col(stat='identity', width=0.6, color="black",
                             position = position_dodge(width=0.6))
                  + ggtitle("Performance Comparison: Metric Overview")
                  + facet_grid("metric ~ class")
                  + coord_flip()
                  + xlab("")
                  + ylab("Score")
                  + scale_y_continuous(limits=[0, 1])
                  + theme_bw()
                  + theme(legend_position="none"))
    # Plot metric results class macro-averaged
    else:
        fig = (ggplot(df, aes("model", "score", fill="model"))
                  + geom_col(stat='identity', width=0.6, color="black",
                             position = position_dodge(width=0.6))
                  + ggtitle("Performance Comparison: Metric Overview")
                  + facet_wrap("metric")
                  + coord_flip()
                  + xlab("")
                  + ylab("Score")
                  + scale_y_continuous(limits=[0, 1])
                  + theme_bw()
                  + theme(legend_position="none"))

    # Store figure to disk
    filename = "plot.comparison.beside"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".png"
    fig.save(filename=filename, path=out_path, width=18, height=9, dpi=300)

#-----------------------------------------------------#
#            Evaluation Comparison - Gain             #
#-----------------------------------------------------#
def evalby_gain(df, out_path, suffix=None):
    # Remove confusion matrix from dataframe
    df = df[~df["metric"].isin(["TN", "FN", "FP", "TP"])]

    # Define gain computation function
    def compute_gain(row, template):
        # Identify current metric
        m = row["metric"]
        # Obtain class-wise divisor
        if "class" in row.index:
            c = row["class"]
            div = template.loc[(template["metric"] == m) & \
                               (template["class"] == c)]["score"].values[0]
        # Obtain macro-averaged divisor
        else:
            div = template.loc[template["metric"] == m]["score"].values[0]
        # Compute gain in percentage compared to template model
        row["score"] = (row["score"] / div) - 1.0
        return row

    # Compute percentage gain compared to first model
    first_model = df["model"].iloc[0]
    template = df.loc[df["model"] == first_model]
    df = df.apply(compute_gain, axis=1, args=(template,))

    # Plot gain results class-wise
    if "class" in df.columns:
        fig = (ggplot(df, aes("model", "score", fill="model"))
                  + geom_col(stat='identity', width=0.6, color="black",
                             position = position_dodge(width=0.2))
                  + ggtitle("Performance Gain compared to Model: " + str(first_model))
                  + facet_grid("metric ~ class")
                  + coord_flip()
                  + xlab("")
                  + ylab("Performance Gain in Percent (%)")
                  + theme_bw()
                  + theme(legend_position="none"))
    # Plot gain results class macro-averaged
    else:
        fig = (ggplot(df, aes("model", "score", fill="model"))
                  + geom_col(stat='identity', width=0.6, color="black",
                             position = position_dodge(width=0.2))
                  + ggtitle("Performance Gain compared to Model: " + str(first_model))
                  + facet_wrap("metric")
                  + coord_flip()
                  + xlab("")
                  + ylab("Performance Gain in Percent (%)")
                  + theme_bw()
                  + theme(legend_position="none"))

    # Store figure to disk
    filename = "plot.comparison.gain"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".png"
    fig.save(filename=filename, path=out_path, width=18, height=9, dpi=300)

    # Return gain dataframe
    return df
