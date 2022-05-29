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

#-----------------------------------------------------#
#              Evaluation - Plot Fitting              #
#-----------------------------------------------------#
def evaluate_fitting(keras_history, out_path, monitor=["loss"],
                     prefix_split=".", suffix=None):
    """ blabla

    Args:
        monitor (list of str):      List of metrics which should be visualized in the fitting plot.
        suffix (str):               Special suffix to add in the created figure filename.

    """
    # Convert to pandas dataframe
    dt = pd.DataFrame.from_dict(keras_history, orient="columns")

    # Identify all selected colums
    selected_cols = []
    for key in keras_history:
        for m in monitor:
            if m in key:
                selected_cols.append(key)
                break

    # Add epoch column
    dt["epoch"] = dt.index + 1
    # Melt dataframe
    dt_melted = dt.melt(id_vars=["epoch"],
                        value_vars=selected_cols,
                        var_name="metric",
                        value_name="score")

    # Handle special prefix tags (if split-able by '.')
    if prefix_split is not None:
        for c in selected_cols:
            valid_split = True
            if prefix_split not in c:
                valid_split = False
                break

        if valid_split:
            dt_melted[["prefix", "metric"]] = dt_melted["metric"].str.split(".",
                                                        expand=True)

    # Preprocess dataframe
    dt_melted["subset"] = np.where(dt_melted["metric"].str.startswith("val_"),
                                   "validation", "training")
    dt_melted["metric"] = dt_melted["metric"].apply(remove_val_tag)

    # Plot results via plotnine
    fig = (ggplot(dt_melted, aes("epoch", "score", color="subset"))
               + geom_line(size=1)
               + ggtitle("Fitting Curve during Training Process")
               + xlab("Epoch")
               + ylab("Score")
               + scale_colour_discrete(name="Subset")
               + theme_bw()
               + theme(subplots_adjust={'wspace':0.15}))

    if prefix_split is not None and valid_split:
        fig += facet_grid("prefix ~ metric")
    else : fig += facet_wrap("metric", scales="free_y")

    # Store figure to disk
    fig.save(filename="plot.fitting_course." + str(suffix) + ".png",
             path=out_path, dpi=200, limitsize=False)

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
def remove_val_tag(x):
    if x.startswith("val_") : return x[4:]
    else : return x
