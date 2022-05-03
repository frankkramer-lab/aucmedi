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
#                    Documentation                    #
#-----------------------------------------------------#
""" Library of implemented Subfunctions in AUCMEDI.

    A Subfunction is a preprocessing method, which is automatically applied on all samples
    if provided to a [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator].

    Image preprocessing is defined as a method or technique which modify the image before passing it to the neural network model.
    The aim of preprocessing methods is to extensively increase performance due to simplification of information.
    Common preprocessing methods range from intensity value normalization to image resizing.

    ???+ info
        The DataGenerator applies the list of Subfunctions **sequentially** on the data set.

    The Subfunctions [Resize][aucmedi.data_processing.subfunctions.resize] and
    [Standardize][aucmedi.data_processing.subfunctions.standardize] are integrated into the
    DataGenerator base due to its requirement in any medical image classification pipeline.

    ???+ example "Example"
        ```python
        # Import Subfunctions
        from aucmedi.data_processing.subfunctions import *

        # Select desired Subfunctions
        sf_crop = Crop(shape=(128, 164), mode="center")
        sf_padding = Padding(mode="square")
        # Pack them into a list
        sf_list = [sf_crop, sf_padding]                 # Subfunctions will be applied in provided list order

        # Pass the list to the DataGenerator
        train_gen = DataGenerator(samples=index_list,
                                  path_imagedir="my_images/",
                                  labels=class_ohe,
                                  resize=(512,512),                # Call the integrated resize Subfunction
                                  standardize_mode="grayscale",    # Call the integrated standardize Subfunction
                                  subfunctions=sf_list)            # Pass desired Subfunctions
        ```

    Subfunctions are based on the abstract base class [Subfunction_Base][aucmedi.data_processing.subfunctions.sf_base.Subfunction_Base],
    which allow simple integration of custom preprocessing methods.
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.data_processing.subfunctions.standardize import Standardize
from aucmedi.data_processing.subfunctions.resize import Resize
from aucmedi.data_processing.subfunctions.padding import Padding
from aucmedi.data_processing.subfunctions.crop import Crop
from aucmedi.data_processing.subfunctions.color_constancy import ColorConstancy
from aucmedi.data_processing.subfunctions.clip import Clip
from aucmedi.data_processing.subfunctions.chromer import Chromer
