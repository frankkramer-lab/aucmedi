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
                     suffix=None):
    """ Function for dataset evaluation (descriptive statistics).
    """

    # Generate barplot
    df_cf = evalby_barplot(labels, out_path, class_names, suffix)

    # Generate heatmap
    evalby_heatmap(samples, labels, out_path, class_names, suffix)

    # Return table with class distribution
    return df_cf

#-----------------------------------------------------#
#             Dataset Analysis - Barplot              #
#-----------------------------------------------------#
def evalby_barplot(labels, out_path, class_names, suffix=None):
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

    # Plot class frequency results
    fig = (ggplot(df_cf, aes("class", "class_perc", fill="class",
                             label=class_freq))
               + geom_bar(stat="identity", color="black")
               + geom_text(nudge_y=3)
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
