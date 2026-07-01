# ==============================================================================#
#  Author:       Fabian Wehr                                                   #
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Internal libraries
from aucmedi import BatchGenerator, create_batch_loader
from aucmedi.data_processing.io_loader import numpy_loader


# -----------------------------------------------------#
#              Unittest: Batch Generator              #
# -----------------------------------------------------#
class BatchGeneratorTEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )
        # Create Grayscale data for 2D
        self.sampleList_gray_2D = []
        for i in range(0, 25):
            img_gray = np.random.rand(16, 16) * 255
            imgGRAY_pillow = Image.fromarray(img_gray.astype(np.uint8))
            index = "image.sample_" + str(i) + ".GRAY.png"
            path_sampleGRAY = os.path.join(self.tmp_data.name, index)
            imgGRAY_pillow.save(path_sampleGRAY)
            self.sampleList_gray_2D.append(index)
        # Create RGB data for 2D
        self.sampleList_rgb_2D = []
        for i in range(0, 25):
            img_rgb = np.random.rand(16, 16, 3) * 255
            imgRGB_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            imgRGB_pillow.save(path_sampleRGB)
            self.sampleList_rgb_2D.append(index)
        # Create Grayscale data for 3D
        self.sampleList_gray_3D = []
        for i in range(0, 25):
            img_gray = np.random.rand(16, 16, 16) * 255
            index = "image.sample_" + str(i) + ".GRAY.npy"
            path_sampleGRAY = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleGRAY, img_gray)
            self.sampleList_gray_3D.append(index)
        # Create RGB data for 3D
        self.sampleList_rgb_3D = []
        for i in range(0, 25):
            img_rgb = np.random.rand(16, 16, 16, 3) * 255
            index = "image.sample_" + str(i) + ".RGB.npy"
            path_sampleRGB = os.path.join(self.tmp_data.name, index)
            np.save(path_sampleRGB, img_rgb)
            self.sampleList_rgb_3D.append(index)

        # Create classification labels
        self.labels_ohe = np.zeros((25, 4), dtype=np.uint8)
        for i in range(0, 25):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create metadata
        self.metadata = np.zeros((25, 10), dtype=np.uint8)
        for i in range(0, 25):
            class_index = np.random.randint(0, 10)
            self.metadata[i][class_index] = 1

        # Create sample weights
        self.sample_weights = np.random.rand(25).astype(np.float32)

    # -------------------------------------------------#
    #           Initialization Functionality          #
    # -------------------------------------------------#
    def test_BASE_create(self):
        batch_gen = BatchGenerator(self.sampleList_rgb_2D, self.tmp_data.name)
        self.assertIsInstance(batch_gen, BatchGenerator)

    def test_BASE_flags(self):
        # No labels / metadata / weights
        batch_gen = BatchGenerator(self.sampleList_rgb_2D, self.tmp_data.name)
        self.assertFalse(batch_gen.has_labels)
        self.assertFalse(batch_gen.has_metadata)
        self.assertFalse(batch_gen.has_sample_weights)

        # With labels
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, labels=self.labels_ohe
        )
        self.assertTrue(batch_gen.has_labels)

        # With metadata
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, metadata=self.metadata
        )
        self.assertTrue(batch_gen.has_metadata)

        # With sample weights
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            sample_weights=self.sample_weights,
        )
        self.assertTrue(batch_gen.has_sample_weights)

    # Pytorch Integration
    def test_BASE_pytorch(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, batch_size=8
        )
        # Must be a PyTorch Dataset
        self.assertIsInstance(batch_gen, Dataset)

        # __len__ returns batch count, not sample count
        expected_batches = math.ceil(len(self.sampleList_rgb_2D) / 8)
        self.assertEqual(len(batch_gen), expected_batches)

        # __getitem__ without labels returns the images array directly (not a tuple)
        batch = batch_gen[0]
        self.assertIsInstance(batch, np.ndarray)
        self.assertEqual(batch.ndim, 4)  # (B, C, H, W)

        # With labels: 2-tuple (images, labels)
        batch_gen_labeled = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            batch_size=8,
        )
        batch_labeled = batch_gen_labeled[0]
        self.assertEqual(len(batch_labeled), 2)
        self.assertIsInstance(batch_labeled[0], np.ndarray)
        self.assertIsInstance(batch_labeled[1], np.ndarray)

        # With sample weights: 3-tuple (images, labels, weights)
        batch_gen_weighted = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            sample_weights=self.sample_weights,
            batch_size=8,
        )
        batch_weighted = batch_gen_weighted[0]
        self.assertEqual(len(batch_weighted), 3)

    def test_BASE_set_length(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, batch_size=8
        )
        original_len = len(batch_gen)
        batch_gen.set_length(2)
        self.assertEqual(len(batch_gen), 2)
        batch_gen.reset_length()
        self.assertEqual(len(batch_gen), original_len)

    # -------------------------------------------------#
    #        Application Functionality for 2D         #
    # -------------------------------------------------#
    # Usage: Grayscale without Labels — channel-first (B, C, H, W)
    def test_RUN_2D_GRAYSCALE_noLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_gray_2D,
            self.tmp_data.name,
            grayscale=True,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertIsInstance(batch, np.ndarray)
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 1)    # channel axis = 1 (grayscale)
            self.assertEqual(batch.shape[2], 224)
            self.assertEqual(batch.shape[3], 224)

    # Usage: RGB without Labels — channel-first (B, C, H, W)
    def test_RUN_2D_RGB_noLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            grayscale=False,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertIsInstance(batch, np.ndarray)
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 3)    # channel axis = 1 (RGB)
            self.assertEqual(batch.shape[2], 224)
            self.assertEqual(batch.shape[3], 224)

    # Usage: With Labels
    def test_RUN_2D_withLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            grayscale=False,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertEqual(len(batch), 2)
            # Image batch: (B, C, H, W)
            self.assertEqual(batch[0].ndim, 4)
            # Label batch: (B, n_classes)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])
            # Batch sizes are consistent
            self.assertEqual(batch[0].shape[0], batch[1].shape[0])

    # Usage: With Sample Weights
    def test_RUN_2D_withWeights(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            sample_weights=self.sample_weights,
            grayscale=False,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertEqual(len(batch), 3)
            b = batch[0].shape[0]
            self.assertEqual(batch[2].shape[0], b)   # weights align with batch

    # -------------------------------------------------#
    #        Application Functionality for 3D         #
    # -------------------------------------------------#
    # Usage: Grayscale without Labels — channel-first (B, C, D, H, W)
    def test_RUN_3D_GRAYSCALE_noLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_gray_3D,
            self.tmp_data.name,
            grayscale=True,
            two_dim=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
            batch_size=4,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertIsInstance(batch, np.ndarray)
            self.assertEqual(batch.ndim, 5)
            self.assertEqual(batch.shape[1], 1)    # channel axis = 1 (grayscale)
            self.assertEqual(batch.shape[2], 16)
            self.assertEqual(batch.shape[3], 16)
            self.assertEqual(batch.shape[4], 16)

    # Usage: RGB without Labels
    def test_RUN_3D_RGB_noLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_3D,
            self.tmp_data.name,
            grayscale=False,
            two_dim=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
            batch_size=4,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertIsInstance(batch, np.ndarray)
            self.assertEqual(batch.ndim, 5)
            self.assertEqual(batch.shape[1], 3)    # channel axis = 1 (RGB)

    # Usage: With Labels
    def test_RUN_3D_withLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_3D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            two_dim=False,
            grayscale=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
            batch_size=4,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].ndim, 5)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])
            self.assertEqual(batch[0].shape[0], batch[1].shape[0])

    # -------------------------------------------------#
    #     Application Functionality with Metadata     #
    # -------------------------------------------------#
    # Usage: Metadata for inference — the 1-tuple is unwrapped to (images, metadata)
    def test_RUN_Metadata_noLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            metadata=self.metadata,
            grayscale=False,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            # len-1 batch is unwrapped: returns (images_arr, metadata_arr) directly
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 2)
            # Image batch is channel-first
            self.assertEqual(batch[0].ndim, 4)
            self.assertEqual(batch[0].shape[1], 3)
            # Metadata batch: (B, meta_variables)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.metadata.shape[1])

    # Usage: Metadata for training
    def test_RUN_Metadata_withLabel(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            metadata=self.metadata,
            grayscale=False,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            # 2-tuple: ((images, metadata), labels)
            self.assertEqual(len(batch), 2)
            self.assertIsInstance(batch[0], tuple)
            self.assertEqual(batch[0][0].ndim, 4)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])

    # -------------------------------------------------#
    #                 Multi-Processing                #
    # -------------------------------------------------#
    def test_MP(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            grayscale=False,
            threads=2,
            batch_size=8,
        )
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].ndim, 4)
            self.assertEqual(batch[1].ndim, 2)

    # -------------------------------------------------#
    #             Beforehand Preprocessing            #
    # -------------------------------------------------#
    def test_PrepareImages(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            prepare_images=True,
            grayscale=False,
            batch_size=8,
        )
        preprocessed_images = os.listdir(batch_gen.prepare_dir)
        self.assertEqual(len(preprocessed_images), len(self.sampleList_rgb_2D))
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].ndim, 4)
        shutil.rmtree(batch_gen.prepare_dir)

    def test_PrepareImages_MP(self):
        batch_gen = BatchGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            prepare_images=True,
            grayscale=False,
            threads=2,
            batch_size=8,
        )
        preprocessed_images = os.listdir(batch_gen.prepare_dir)
        self.assertEqual(len(preprocessed_images), len(self.sampleList_rgb_2D))
        for i in range(0, len(batch_gen)):
            batch = batch_gen[i]
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].ndim, 4)
        shutil.rmtree(batch_gen.prepare_dir)

    # -------------------------------------------------#
    #              WrapperLoader Integration          #
    # -------------------------------------------------#
    def test_WrapperLoader_noLabel(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            grayscale=False,
            batch_size=8,
        )
        for batch in loader:
            # No labels: WrapperLoader yields a bare tensor (length-1 unwrapped)
            self.assertIsInstance(batch, torch.Tensor)
            # Channel-first: (B, C, H, W)
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 3)

    def test_WrapperLoader_withLabel(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            grayscale=False,
            batch_size=8,
        )
        for batch in loader:
            self.assertEqual(len(batch), 2)
            self.assertIsInstance(batch[0], torch.Tensor)
            self.assertIsInstance(batch[1], torch.Tensor)
            self.assertEqual(batch[0].shape[0], batch[1].shape[0])
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])
