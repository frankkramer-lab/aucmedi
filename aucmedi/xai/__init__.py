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
""" Core interface for XAI in AUCMEDI: [aucmedi.xai.decoder.xai_decoder][]

???+ info "XAI Methods"
    The XAI Decoder can be run with different XAI methods as backbone.

    A list of all implemented methods and their keys can be found here: <br>
    [aucmedi.xai.methods][]

???+ example "Example"
    ```python
    # Create a DataGenerator for data I/O
    datagen = DataGenerator(samples[:3], "images_xray/", labels=None, resize=(299, 299))

    # Get a model
    model = NeuralNetwork(n_labels=3, channels=3, architecture="Xception",
                           input_shape=(299,299))
    model.load("model.xray.hdf5")

    # Make some predictions
    preds = model.predict(datagen)

    # Compute XAI heatmaps via Grad-CAM (resulting heatmaps are stored in out_path)
    xai_decoder(datagen, model, preds, method="gradcam", out_path="xai.xray_gradcam")
    ```
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# Import XAI functionalities
from aucmedi.xai.methods import xai_dict
from aucmedi.xai.decoder import xai_decoder
