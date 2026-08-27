# ==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2026 IT-Infrastructure for Translational Medical Research,    #
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import unittest
import tempfile
import os
from PIL import Image
import numpy as np
from torch.nn import CrossEntropyLoss

# Internal libraries
from aucmedi.neural_network.loss_functions import *
from aucmedi import *


# -----------------------------------------------------#
#               Unittest: Loss Functions              #
# -----------------------------------------------------#
class LossfunctionsTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )
        # Create data
        self.sampleList = []
        for i in range(0, 1):
            img = np.random.rand(32, 32, 3) * 255
            img_pillow = Image.fromarray(img.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            path_sample = os.path.join(self.tmp_data.name, index)
            img_pillow.save(path_sample)
            self.sampleList.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((1, 4), dtype=np.uint8)
        for i in range(0, 1):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1
        # Create Data Loader
        self.dataloader = create_data_loader(
            self.sampleList,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(32, 32),
            grayscale=False,
            batch_size=1,
        )

    # -------------------------------------------------#
    #                 Categorical Crossentropy         #
    # -------------------------------------------------#
    def test_Torch(self):
        loss_ce = CrossEntropyLoss()
        model = NeuralNetwork(
            n_labels=4,
            channels=3,
            loss=loss_ce,
            input_resolution=(32, 32),
        )
        model.train(self.dataloader, epochs=1)

    # -------------------------------------------------#
    #                Focal Loss: Binary               #
    # -------------------------------------------------#
    def test_BinaryFocalLoss(self):
        lf = BinaryFocalLoss(alpha=0.25, gamma=2)
        model = NeuralNetwork(
            n_labels=4, channels=3, loss=lf, input_resolution=(32, 32)
        )
        model.train(self.dataloader, epochs=1)

    # -------------------------------------------------#
    #             Focal Loss: Categorical             #
    # -------------------------------------------------#
    def test_MultiClassFocalLoss(self):
        lf = MultiClassFocalLoss(alpha=[0.25, 0.25, 0.5, 4.0], gamma=2)
        model = NeuralNetwork(
            n_labels=4, channels=3, loss=lf, input_resolution=(32, 32)
        )
        model.train(self.dataloader, epochs=1)

    # -------------------------------------------------#
    #             Focal Loss: Multi-Label             #
    # -------------------------------------------------#
    def test_MultiLabelFocalLoss(self):
        lf = MultiLabelFocalLoss(alpha=[0.25, 0.25, 0.5, 4.0], gamma=2)
        model = NeuralNetwork(
            n_labels=4, channels=3, loss=lf, input_resolution=(32, 32)
        )
        model.train(self.dataloader, epochs=1)
