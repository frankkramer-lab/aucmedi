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
#            Evaluation - Plot Performance            #
#-----------------------------------------------------#
def evaluate_performance(preds,
                         labels,
                         out_path,
                         show=False,
                         class_names=None,
                         multi_label=False,
                         metrics_threshold=0.5,
                         suffix=None,
                         store_csv=True,
                         plot_barplot=True,
                         plot_confusion_matrix=True,
                         plot_roc_curve=True):
    """ Function for automatic performance evaluation based on model predictions.

    ???+ example
        ```python
        # Import libraries
        from aucmedi import *
        from aucmedi.evaluation import *

        # Load data
        ds = input_interface(interface="csv",                       # Interface type
                             path_imagedir="dataset/images/",
                             path_data="dataset/annotations.csv",
                             ohe=False, col_sample="ID", col_class="diagnosis")
        (samples, class_ohe, nclasses, class_names, image_format) = ds

        # Initialize model
        model = NeuralNetwork(n_labels=8, channels=3, architecture="2D.ResNet50")

        # Do some predictions
        datagen_test = DataGenerator(samples, "dataset/images/", labels=None,
                                     resize=model.meta_input, standardize_mode=model.meta_standardize)
        preds = model.predict(datagen_test)

        # Pass predictions to evaluation function
        evaluate_performance(preds, class_ohe, out_path="./", class_names=class_names)
        ```

    Created files in directory of `out_path`:

    - with `store_csv`: "metrics.performance.csv"
    - with `plot_barplot`: "plot.performance.barplot.png"
    - with `plot_confusion_matrix`: "plot.performance.confusion_matrix.png"
    - with `plot_roc_curve`: "plot.performance.roc.png"

    ???+ info "Preview for Bar Plot"
        ![Evaluation_Performance_Barplot](../../images/evaluation.plot.performance.barplot.png)

        Predictions based on [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
        utilizing a DenseNet121.

    ???+ info "Preview for Confusion Matrix"
        ![Evaluation_Performance_ConfusionMatrix](../../images/evaluation.plot.performance.confusion_matrix.png)

        Predictions based on [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
        utilizing a DenseNet121.

    ???+ info "Preview for ROC Curve"
        ![Evaluation_Performance_ROCcurve](../../images/evaluation.plot.performance.roc.png)

        Predictions based on [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
        utilizing a DenseNet121.

    Args:
        preds (numpy.ndarray):          A NumPy array of predictions formatted with shape (n_samples, n_labels). Provided by
                                        [NeuralNetwork][aucmedi.neural_network.model].
        labels (numpy.ndarray):         Classification list with One-Hot Encoding. Provided by
                                        [input_interface][aucmedi.data_processing.io_data.input_interface].
        out_path (str):                 Path to directory in which plotted figures are stored.
        show (bool):                    Option, whether to also display the generated charts.
        class_names (list of str):      List of names for corresponding classes. Used for evaluation. Provided by
                                        [input_interface][aucmedi.data_processing.io_data.input_interface].
                                        If not provided (`None` provided), class indices will be used.
        multi_label (bool):             Option, whether task is multi-label based (has impact on evaluation).
        metrics_threshold (float):      Only required if 'multi_label==True`. Threshold value if prediction is positive.
                                        Used in metric computation for CSV and bar plot.
        suffix (str):                   Special suffix to add in the created figure filename.
        store_csv (bool):               Option, whether to generate a CSV file containing various metrics.
        plot_barplot (bool):            Option, whether to generate a bar plot of various metrics.
        plot_confusion_matrix (bool):   Option, whether to generate a confusion matrix plot.
        plot_roc_curve (bool):          Option, whether to generate a ROC curve plot.

    Returns:
        metrics (pandas.DataFrame):     Dataframe containing all computed metrics (except ROC).
    """
    # Identify number of labels
    n_labels = labels.shape[-1]
    # Identify prediction threshold
    if multi_label : threshold = metrics_threshold
    else : threshold = None

    # Compute metrics
    metrics = compute_metrics(preds, labels, n_labels, threshold)
    cm = compute_confusion_matrix(preds, labels, n_labels)
    fpr_list, tpr_list = compute_roc(preds, labels, n_labels)

    # Rename columns in metrics dataframe
    class_mapping = {}
    if class_names is not None:
        for c in range(len(class_names)):
            class_mapping[c] = class_names[c]
        metrics["class"].replace(class_mapping, inplace=True)
    if class_names is None : metrics["class"] = pd.Categorical(metrics["class"])

    # Store metrics to CSV
    if store_csv:
        evalby_csv(metrics, out_path, class_names, suffix=suffix)

    # Generate bar plot
    if plot_barplot:
        evalby_barplot(metrics, out_path, class_names, show=show, suffix=suffix)

    # Generate confusion matrix plot
    if plot_confusion_matrix and not multi_label:
        evalby_confusion_matrix(cm, out_path, class_names, show=show, suffix=suffix)

    # Generate ROC curve
    if plot_roc_curve:
        evalby_rocplot(fpr_list, tpr_list, out_path, class_names, show=show, suffix=suffix)

    # Return metrics
    return metrics

#-----------------------------------------------------#
#      Evaluation Performance - Confusion Matrix      #
#-----------------------------------------------------#
def evalby_confusion_matrix(confusion_matrix, out_path, class_names,
                            show=False,
                            suffix=None):

    # Convert confusion matrix to a Pandas dataframe
    rawcm = pd.DataFrame(confusion_matrix)
    # Tidy dataframe
    if class_names is None or len(class_names) != confusion_matrix.shape[0]:
        class_names = list(range(0, confusion_matrix.shape[0]))
    rawcm.index = class_names
    rawcm.columns = class_names

    # Preprocess dataframe
    dt = rawcm.div(rawcm.sum(axis=1), axis=0).fillna(0) * 100
    dt = dt.round(decimals=2)
    dt.reset_index(drop=False, inplace=True)
    dt = dt.melt(id_vars=["index"], var_name="pd", value_name="score")
    dt.rename(columns={"index": "gt"}, inplace=True)

    # Generate confusion matrix
    fig = (ggplot(dt, aes("pd", "gt", fill="score"))
                  + geom_tile(color="white", size=1.5)
                  + geom_text(aes("pd", "gt", label="score"), color="black")
                  + ggtitle("Performance Evaluation: Confusion Matrix")
                  + xlab("Prediction")
                  + ylab("Ground Truth")
                  + scale_fill_gradient(low="white", high="royalblue",
                                        limits=[0, 100])
                  + guides(fill = guide_colourbar(title="%",
                                                  barwidth=10,
                                                  barheight=50))
                  + theme_bw()
                  + theme(axis_text_x = element_text(angle = 45, vjust = 1,
                                                     hjust = 1)))

    # Store figure to disk
    filename = "plot.performance.confusion_matrix"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".png"
    fig.save(filename=filename, path=out_path, width=10, height=9, dpi=200)

    # Plot figure
    if show : print(fig)

#-----------------------------------------------------#
#          Evaluation Performance - Barplots          #
#-----------------------------------------------------#
def evalby_barplot(metrics, out_path, class_names, show=False, suffix=None):
    # Remove confusion matrix from metric dataframe
    df_metrics = metrics[~metrics["metric"].isin(["TN", "FN", "FP", "TP"])]
    df_metrics["class"] = pd.Categorical(df_metrics["class"])

    # Generate metric results
    fig = (ggplot(df_metrics, aes("class", "score", fill="class"))
              + geom_col(stat='identity', width=0.6, color="black",
                         position = position_dodge(width=0.6))
              + ggtitle("Performance Evaluation: Metric Overview")
              + facet_wrap("metric")
              + coord_flip()
              + xlab("")
              + ylab("Score")
              + scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.1))
              + scale_fill_discrete(name="Classes")
              + theme_bw())

    # Store figure to disk
    filename = "plot.performance.barplot"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".png"
    fig.save(filename=filename, path=out_path, width=12, height=9, dpi=200)

    # Plot figure
    if show : print(fig)

#-----------------------------------------------------#
#          Evaluation Performance - ROC plot          #
#-----------------------------------------------------#
def evalby_rocplot(fpr_list, tpr_list, out_path, class_names, show=False, suffix=None):
    # Initialize result dataframe
    df_roc = pd.DataFrame(data=[fpr_list, tpr_list], dtype=object)
    # Preprocess dataframe
    df_roc = df_roc.transpose()
    df_roc = df_roc.apply(pd.Series.explode)
    # Rename columns
    class_mapping = {}
    if class_names is not None:
        for c in range(len(class_names)):
            class_mapping[c] = class_names[c]
        df_roc.rename(index=class_mapping, inplace=True)
    df_roc = df_roc.reset_index()
    df_roc.rename(columns={"index": "class", 0: "FPR", 1: "TPR"}, inplace=True)
    df_roc["class"] = pd.Categorical(df_roc["class"])
    # Convert from object to float
    df_roc["FPR"] = df_roc["FPR"].astype(float)
    df_roc["TPR"] = df_roc["TPR"].astype(float)

    # Generate roc results
    fig = (ggplot(df_roc, aes("FPR", "TPR", color="class"))
               + geom_line(size=1.0)
               + geom_abline(intercept=0, slope=1, color="black",
                             linetype="dashed")
               + ggtitle("Performance Evaluation: ROC Curves")
               + xlab("False Positive Rate")
               + ylab("True Positive Rate")
               + scale_x_continuous(limits=[0, 1], breaks=np.arange(0,1.1,0.1))
               + scale_y_continuous(limits=[0, 1], breaks=np.arange(0,1.1,0.1))
               + scale_color_discrete(name="Classes")
               + theme_bw())

    # Store figure to disk
    filename = "plot.performance.roc"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".png"
    fig.save(filename=filename, path=out_path, width=10, height=9, dpi=200)

    # Plot figure
    if show : print(fig)

#-----------------------------------------------------#
#          Evaluation Performance - CSV file          #
#-----------------------------------------------------#
def evalby_csv(metrics, out_path, class_names, suffix=None):
    # Obtain filename to
    filename = "metrics.performance"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".csv"
    path_csv = os.path.join(out_path, filename)

    # Store file to disk
    metrics.to_csv(path_csv, index=False)
