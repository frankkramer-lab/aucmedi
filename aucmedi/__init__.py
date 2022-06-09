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
""" This is the API reference for the AUCMEDI framework.

Build your state-of-the-art medical image classification pipeline with the 3 AUCMEDI pillars:

!!! info "Pillars of AUCMEDI"
    | Pillar                                                                       | Description                                                    |
    | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
    | #1: [input_interface()][aucmedi.data_processing.io_data.input_interface]  | Obtaining general information from the dataset.                   |
    | #2: [NeuralNetwork][aucmedi.neural_network.model.NeuralNetwork]         | Building the deep learning model.                                 |
    | #3: [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator] | Powerful interface for loading any images/volumes into the model. |


???+ example "A typical AUCMEDI pipeline"
    ```python
    # AUCMEDI library
    from aucmedi import *

    # Pillar #1: Initialize input data reader
    ds = input_interface(interface="csv",
                         path_imagedir="dataset/images/",
                         path_data="dataset/classes.csv",
                         ohe=False, col_sample="ID", col_class="diagnosis")
    (index_list, class_ohe, nclasses, class_names, image_format) = ds

    # Pillar #2: Initialize a DenseNet121 model with ImageNet weights
    model = NeuralNetwork(n_labels=nclasses, channels=3,
                           architecture="2D.DenseNet121",
                           pretrained_weights=True)

    # Pillar #3: Initialize training Data Generator for first 1000 samples
    train_gen = DataGenerator(samples=index_list[:1000],
                              path_imagedir="dataset/images/",
                              labels=class_ohe[:1000],
                              image_format=image_format)
    # Run model training with Transfer Learning
    model.train(train_gen, epochs=20, transfer_learning=True)

    # Pillar #3: Initialize testing Data Generator for 500 samples
    test_gen = DataGenerator(samples=index_list[1000:1500],
                             path_imagedir="dataset/images/",
                             labels=None,
                             image_format=image_format)
    # Run model inference for unknown samples
    preds = model.predict(test_gen)
    ```
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from aucmedi.data_processing.io_data import input_interface
from aucmedi.data_processing.data_generator import DataGenerator
from aucmedi.data_processing.augmentation import ImageAugmentation, \
                                                 VolumeAugmentation, \
                                                 BatchgeneratorsAugmentation
from aucmedi.neural_network.model import NeuralNetwork
