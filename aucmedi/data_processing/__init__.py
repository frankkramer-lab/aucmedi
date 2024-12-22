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
#                    Documentation                    #
#-----------------------------------------------------#
""" The data processing is a important part of any AUCMEDI pipeline and represented by two core pillars.

- The [input_interface][aucmedi.data_processing.io_data.input_interface] function provides important information on the
    dataset.
- The [DataGenerator][aucmedi.data_processing.data_generator.DataGenerator] class is a
powerful interface handling all data processing in AUCMEDI.

???+ info "Pillars of AUCMEDI"
    - [aucmedi.data_processing.io_data.input_interface][]
    - [aucmedi.data_processing.data_generator.DataGenerator][]
    - [aucmedi.neural_network.model.NeuralNetwork][]

With an initialized Neural Network instance, it is possible to run training and predictions.

???+ example
    ```python
    # Import
    from aucmedi import *

    # Initialize input data reader
    ds = input_interface(interface="csv",
                         path_imagedir="dataset/images/",
                         path_data="dataset/annotations.csv",
                         ohe=False, col_sample="ID", col_class="diagnosis")
    (samples, class_ohe, nclasses, class_names, image_format) = ds

    # Initialize model
    model = NeuralNetwork(n_labels=nclasses, channels=3, architecture="2D.ResNet50")

    # Do some training
    datagen_train = DataGenerator(samples[:100], "images_dir/", labels=class_ohe[:100], image_format=image_format,
                                  resize=model.meta_input, standardize_mode=model.meta_standardize)
    model.train(datagen_train, epochs=50)

    # Do some predictions
    datagen_test = DataGenerator(samples[100:150], "images_dir/", labels=None, image_format=image_format,
                                 resize=model.meta_input, standardize_mode=model.meta_standardize)
    preds = model.predict(datagen_test)
    ```
"""
