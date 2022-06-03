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
                        suffix=None,
                        store_csv=True,
                        plot_barplot=True):
    """ Function for performance comparison evaluation based on predictions from multiple models.

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
            metrics_avg = metrics.groupby(["metric", "model"]).mean()
            metrics = metrics_avg.reset_index()

        # Append to dataframe list
        df_list.append(metrics)
    # Combine dataframes
    df_merged = pd.concat(df_list, axis=0, ignore_index=True)

    # Generate comparison beside plot
    evalby_beside(df_merged, out_path, suffix)

    # Generate comparison gain plot
    evalby_gain(df_merged, out_path, suffix)

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
