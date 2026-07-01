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
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Internal libraries
from aucmedi import BatchGenerator, create_batch_loader
from aucmedi.data_processing.io_loader import numpy_loader
from aucmedi.data_processing.wrapper_loader import WrapperLoader


# -----------------------------------------------------#
#              Unittest: Wrapper Loader               #
# -----------------------------------------------------#
class WrapperLoaderTEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        self.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )
        # 2D grayscale
        self.sampleList_gray_2D = []
        for i in range(0, 25):
            img = np.random.rand(16, 16) * 255
            pillow = Image.fromarray(img.astype(np.uint8))
            index = f"image.sample_{i}.GRAY.png"
            pillow.save(os.path.join(self.tmp_data.name, index))
            self.sampleList_gray_2D.append(index)
        # 2D RGB
        self.sampleList_rgb_2D = []
        for i in range(0, 25):
            img = np.random.rand(16, 16, 3) * 255
            pillow = Image.fromarray(img.astype(np.uint8))
            index = f"image.sample_{i}.RGB.png"
            pillow.save(os.path.join(self.tmp_data.name, index))
            self.sampleList_rgb_2D.append(index)
        # 3D grayscale
        self.sampleList_gray_3D = []
        for i in range(0, 25):
            img = np.random.rand(16, 16, 16) * 255
            index = f"image.sample_{i}.GRAY.npy"
            np.save(os.path.join(self.tmp_data.name, index), img)
            self.sampleList_gray_3D.append(index)
        # 3D RGB
        self.sampleList_rgb_3D = []
        for i in range(0, 25):
            img = np.random.rand(16, 16, 16, 3) * 255
            index = f"image.sample_{i}.RGB.npy"
            np.save(os.path.join(self.tmp_data.name, index), img)
            self.sampleList_rgb_3D.append(index)

        self.labels_ohe = np.zeros((25, 4), dtype=np.uint8)
        for i in range(0, 25):
            self.labels_ohe[i][np.random.randint(0, 4)] = 1

        self.metadata = np.zeros((25, 10), dtype=np.uint8)
        for i in range(0, 25):
            self.metadata[i][np.random.randint(0, 10)] = 1

        self.sample_weights = np.random.rand(25).astype(np.float32)

    # -------------------------------------------------#
    #           Initialization Functionality          #
    # -------------------------------------------------#
    def test_BASE_create(self):
        loader = create_batch_loader(self.sampleList_rgb_2D, self.tmp_data.name)
        self.assertIsInstance(loader, WrapperLoader)
        self.assertIsInstance(loader, DataLoader)

    def test_BASE_direct_construction(self):
        batch_gen = BatchGenerator(self.sampleList_rgb_2D, self.tmp_data.name)
        loader = WrapperLoader(batch_gen)
        self.assertIsInstance(loader, WrapperLoader)

    def test_BASE_flags_proxied(self):
        # Flags from BatchGenerator must be visible on WrapperLoader
        loader = create_batch_loader(self.sampleList_rgb_2D, self.tmp_data.name)
        self.assertFalse(loader.has_labels)
        self.assertFalse(loader.has_metadata)
        self.assertFalse(loader.has_sample_weights)

        loader_l = create_batch_loader(
            self.sampleList_rgb_2D, self.tmp_data.name, labels=self.labels_ohe
        )
        self.assertTrue(loader_l.has_labels)

        loader_m = create_batch_loader(
            self.sampleList_rgb_2D, self.tmp_data.name, metadata=self.metadata
        )
        self.assertTrue(loader_m.has_metadata)

        loader_w = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            sample_weights=self.sample_weights,
        )
        self.assertTrue(loader_w.has_sample_weights)

    def test_BASE_properties(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            metadata=self.metadata,
        )
        self.assertEqual(list(loader.samples), self.sampleList_rgb_2D)
        self.assertTrue(np.array_equal(loader.labels, self.labels_ohe))
        self.assertTrue(np.array_equal(loader.metadata, self.metadata))

    def test_BASE_len(self):
        batch_size = 8
        loader = create_batch_loader(
            self.sampleList_rgb_2D, self.tmp_data.name, batch_size=batch_size
        )
        expected = math.ceil(len(self.sampleList_rgb_2D) / batch_size)
        self.assertEqual(len(loader), expected)

    def test_BASE_set_length(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D, self.tmp_data.name, batch_size=8
        )
        original_len = len(loader)
        loader.set_length(2)
        self.assertEqual(len(loader), 2)
        loader.reset_length()
        self.assertEqual(len(loader), original_len)

    # -------------------------------------------------#
    #       Output Type: torch.Tensor / float32       #
    # -------------------------------------------------#
    def test_TENSOR_type_and_dtype(self):
        # All yielded image tensors must be float32
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            batch_size=8,
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].dtype, torch.float32)
        self.assertIsInstance(batch[1], torch.Tensor)
        self.assertEqual(batch[1].dtype, torch.float32)

    def test_TENSOR_noLabel_bare(self):
        # No labels: bare tensor (not wrapped in a tuple)
        loader = create_batch_loader(
            self.sampleList_rgb_2D, self.tmp_data.name, batch_size=8
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.Tensor)

    def test_TENSOR_withLabel_tuple(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            batch_size=8,
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertIsInstance(batch[1], torch.Tensor)

    def test_TENSOR_withWeights_triple(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            sample_weights=self.sample_weights,
            batch_size=8,
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 3)
        for element in batch:
            self.assertIsInstance(element, torch.Tensor)

    # -------------------------------------------------#
    #        Application Functionality for 2D         #
    # -------------------------------------------------#
    def test_RUN_2D_GRAYSCALE_noLabel(self):
        loader = create_batch_loader(
            self.sampleList_gray_2D,
            self.tmp_data.name,
            grayscale=True,
            batch_size=8,
        )
        for batch in loader:
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.ndim, 4)          # (B, C, H, W)
            self.assertEqual(batch.shape[1], 1)       # grayscale
            self.assertEqual(batch.shape[2], 224)
            self.assertEqual(batch.shape[3], 224)

    def test_RUN_2D_RGB_noLabel(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            grayscale=False,
            batch_size=8,
        )
        for batch in loader:
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 3)       # RGB
            self.assertEqual(batch.shape[2], 224)
            self.assertEqual(batch.shape[3], 224)

    def test_RUN_2D_withLabel(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            grayscale=False,
            batch_size=8,
        )
        for batch in loader:
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].ndim, 4)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])
            self.assertEqual(batch[0].shape[0], batch[1].shape[0])

    # -------------------------------------------------#
    #        Application Functionality for 3D         #
    # -------------------------------------------------#
    def test_RUN_3D_GRAYSCALE_noLabel(self):
        loader = create_batch_loader(
            self.sampleList_gray_3D,
            self.tmp_data.name,
            grayscale=True,
            two_dim=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
            batch_size=4,
        )
        for batch in loader:
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.ndim, 5)          # (B, C, D, H, W)
            self.assertEqual(batch.shape[1], 1)       # grayscale

    def test_RUN_3D_RGB_noLabel(self):
        loader = create_batch_loader(
            self.sampleList_rgb_3D,
            self.tmp_data.name,
            grayscale=False,
            two_dim=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
            batch_size=4,
        )
        for batch in loader:
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(batch.ndim, 5)
            self.assertEqual(batch.shape[1], 3)

    def test_RUN_3D_withLabel(self):
        loader = create_batch_loader(
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
        for batch in loader:
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].ndim, 5)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])

    # -------------------------------------------------#
    #     Application Functionality with Metadata     #
    # -------------------------------------------------#
    def test_RUN_Metadata_noLabel(self):
        # No labels + metadata: unwrapped to (image_tensor, meta_tensor)
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            metadata=self.metadata,
            grayscale=False,
            batch_size=8,
        )
        for batch in loader:
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 2)
            self.assertIsInstance(batch[0], torch.Tensor)
            self.assertEqual(batch[0].ndim, 4)
            self.assertEqual(batch[0].shape[1], 3)
            self.assertIsInstance(batch[1], torch.Tensor)
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.metadata.shape[1])

    def test_RUN_Metadata_withLabel(self):
        # Labels + metadata: ((image_tensor, meta_tensor), label_tensor)
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            metadata=self.metadata,
            grayscale=False,
            batch_size=8,
        )
        for batch in loader:
            self.assertEqual(len(batch), 2)
            self.assertIsInstance(batch[0], tuple)
            self.assertIsInstance(batch[0][0], torch.Tensor)  # images
            self.assertIsInstance(batch[0][1], torch.Tensor)  # metadata
            self.assertIsInstance(batch[1], torch.Tensor)      # labels
            self.assertEqual(batch[1].ndim, 2)
            self.assertEqual(batch[1].shape[1], self.labels_ohe.shape[1])

    # -------------------------------------------------#
    #              Iteration Completeness             #
    # -------------------------------------------------#
    def test_ITER_covers_all_batches(self):
        batch_size = 8
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            batch_size=batch_size,
        )
        expected_batches = math.ceil(len(self.sampleList_rgb_2D) / batch_size)
        actual_batches = sum(1 for _ in loader)
        self.assertEqual(actual_batches, expected_batches)

    def test_ITER_set_length_limits_batches(self):
        loader = create_batch_loader(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            batch_size=8,
        )
        loader.set_length(2)
        actual_batches = sum(1 for _ in loader)
        self.assertEqual(actual_batches, 2)
        loader.reset_length()
