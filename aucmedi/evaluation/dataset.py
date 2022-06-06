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

#-----------------------------------------------------#
#            Evaluation - Dataset Analysis            #
#-----------------------------------------------------#
def evaluate_dataset(samples,
                     labels,
                     out_path,
                     class_names=None,
                     plot_barplot=False,
                     plot_heatmap=False,
                     suffix=None):
    """ Function for dataset evaluation (descriptive statistics).

    ???+ example
        ```python
        # Import libraries
        from aucmedi import *
        from aucmedi.evaluation import *

        # Peak data information via the first pillar of AUCMEDI
        ds = input_interface(interface="csv",                       # Interface type
                             path_imagedir="dataset/images/",
                             path_data="dataset/annotations.csv",
                             ohe=False, col_sample="ID", col_class="diagnosis")
        (samples, class_ohe, nclasses, class_names, image_format) = ds

        # Pass information to the evaluation function
        evaluate_dataset(samples, class_ohe, out_path="./", class_names=class_names)
        ```

    Created files in directory of `out_path`:

    - "plot.dataset.barplot.png"
    - "plot.dataset.heatmap.png"

    ???+ info "Preview for Bar Plot"
        ![Evaluation_Dataset_Barplot](../../images/evaluation.plot.dataset.barplot.png)

        Based on dataset: [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/).

    ???+ info "Preview for Heatmap"
        ![Evaluation_Dataset_Heatmap](../../images/evaluation.plot.dataset.heatmap.png)

        Based on first 50 samples from dataset: [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/).

    Args:
        samples (list of str):              List of sample/index encoded as Strings. Provided by
                                            [input_interface][aucmedi.data_processing.io_data.input_interface].
        labels (numpy.ndarray):             Classification list with One-Hot Encoding. Provided by
                                            [input_interface][aucmedi.data_processing.io_data.input_interface].
        out_path (str):                     Path to directory in which plotted figures are stored.
        class_names (list of str):          List of names for corresponding classes. Used for evaluation. Provided by
                                            [input_interface][aucmedi.data_processing.io_data.input_interface].
                                            If not provided (`None` provided), class indices will be used.
        plot_barplot (bool):                Option, whether to generate a bar plot of class distribution.
        plot_heatmap (bool):                Option, whether to generate a heatmap of class overview. Only recommended for subsets of ~50 samples.
        suffix (str):                       Special suffix to add in the created figure filename.

    Returns:
        df_cf (pandas.DataFrame):           Dataframe containing the class distribution of the dataset.
    """

    # Generate barplot
    df_cf = evalby_barplot(labels, out_path, class_names, plot_barplot, suffix)

    # Generate heatmap
    if plot_heatmap:
        evalby_heatmap(samples, labels, out_path, class_names, suffix)

    # Return table with class distribution
    return df_cf

#-----------------------------------------------------#
#             Dataset Analysis - Barplot              #
#-----------------------------------------------------#
def evalby_barplot(labels, out_path, class_names, plot_barplot, suffix=None):
    # compute class frequency
    cf_list = []
    for c in range(0, labels.shape[1]):
        n_samples = labels.shape[0]
        class_freq = np.sum(labels[:, c])
        if class_names is None : curr_class = str(c)
        else : curr_class = class_names[c]
        class_percentage = np.round(class_freq / n_samples, 2) * 100
        cf_list.append([curr_class, class_freq, class_percentage])

    # Convert class frequency results to dataframe
    df_cf = pd.DataFrame(np.array(cf_list),
                         columns=["class", "class_freq", "class_perc"])
    df_cf["class_perc"] = pd.to_numeric(df_cf["class_perc"])

    if plot_barplot:
        # Plot class frequency results
        fig = (ggplot(df_cf, aes("class", "class_perc", fill="class"))
                   + geom_bar(stat="identity", color="black")
                   + geom_text(aes(label="class_freq"), nudge_y=5)
                   + coord_flip()
                   + ggtitle("Dataset Analysis: Class Distribution")
                   + xlab("Classes")
                   + ylab("Class Frequency (in %)")
                   + scale_y_continuous(limits=[0, 100],
                                        breaks=np.arange(0,110,10))
                   + theme_bw()
                   + theme(legend_position="none"))

        # Store figure to disk
        filename = "plot.dataset.barplot"
        if suffix is not None : filename += "." + str(suffix)
        filename += ".png"
        fig.save(filename=filename, path=out_path, width=10, height=9, dpi=200)

    # Return class table
    return df_cf

#-----------------------------------------------------#
#             Dataset Analysis - Heatmap              #
#-----------------------------------------------------#
def evalby_heatmap(samples, labels, out_path, class_names, suffix=None):
    # Create dataframe
    if class_names is None : df = pd.DataFrame(labels, index=samples)
    else : df = pd.DataFrame(labels, index=samples, columns=class_names)

    # Preprocess dataframe
    df = df.reset_index()
    df_melted = pd.melt(df, id_vars="index", var_name="class",
                        value_name="presence")

    # Plot heatmap
    fig = (ggplot(df_melted, aes("index", "class", fill="presence"))
               + geom_tile()
               + coord_flip()
               + ggtitle("Dataset Analysis: Overview")
               + xlab("Samples")
               + ylab("Classes")
               + scale_fill_gradient(low="white", high="#3399FF")
               + theme_classic()
               + theme(legend_position="none"))

    # Store figure to disk
    filename = "plot.dataset.heatmap"
    if suffix is not None : filename += "." + str(suffix)
    filename += ".png"
    fig.save(filename=filename, path=out_path, width=10, height=9, dpi=200)
