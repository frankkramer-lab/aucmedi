# ==============================================================================#
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
# ==============================================================================#
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import unittest
import numpy as np
import tempfile
from PIL import Image
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader

# Internal libraries
from aucmedi import DataGenerator, NeuralNetwork
from aucmedi.data_processing.io_loader import numpy_loader


# -----------------------------------------------------#
#               Unittest: Data Generator              #
# -----------------------------------------------------#
class DataGeneratorTEST(unittest.TestCase):
    # Create random imaging and classification data
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

    # -------------------------------------------------#
    #           Initialization Functionality          #
    # -------------------------------------------------#
    # Class Creation
    def test_BASE_create(self):
        data_gen = DataGenerator(self.sampleList_rgb_2D, self.tmp_data.name)
        self.assertIsInstance(data_gen, DataGenerator)

    # Pytorch Integration
    def test_BASE_pytorch(self):
        # DataGenerator must be a PyTorch Dataset
        data_gen = DataGenerator(self.sampleList_rgb_2D, self.tmp_data.name)
        self.assertIsInstance(data_gen, Dataset)

        # len() reflects the number of samples
        self.assertEqual(len(data_gen), len(self.sampleList_rgb_2D))

        # __getitem__ returns a 1-tuple; image element is a torch.Tensor
        item = data_gen[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 1)
        self.assertIsInstance(item[0], torch.Tensor)

        # With labels: 2-tuple (input, label), both torch.Tensor
        data_gen_labeled = DataGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, labels=self.labels_ohe
        )
        item_labeled = data_gen_labeled[0]
        self.assertEqual(len(item_labeled), 2)
        self.assertIsInstance(item_labeled[0], torch.Tensor)
        self.assertIsInstance(item_labeled[1], torch.Tensor)

        # With metadata: input element is a (img_tensor, metadata_tensor) tuple
        data_gen_meta = DataGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, metadata=self.metadata
        )
        item_meta = data_gen_meta[0]
        self.assertEqual(len(item_meta), 1)
        self.assertIsInstance(item_meta[0], tuple)
        self.assertIsInstance(item_meta[0][0], torch.Tensor)
        self.assertIsInstance(item_meta[0][1], torch.Tensor)

        # Compatible with DataLoader — collation produces batched tensors
        loader = DataLoader(data_gen_labeled, batch_size=4, shuffle=False)
        self.assertIsInstance(loader, DataLoader)
        batch = next(iter(loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertIsInstance(batch[1], torch.Tensor)
        # Batch dimension is 4
        self.assertEqual(batch[0].shape[0], 4)
        # Label batch shape: (4, n_classes)
        self.assertEqual(batch[1].shape, torch.Size([4, self.labels_ohe.shape[1]]))

        # Without labels: default_collate turns the 1-tuple sample into a
        # single-element list, NOT a bare Tensor -- this differs from
        # WrapperLoader/BatchGenerator, which unwrap it themselves.
        # (Regression: model.predict() used to crash on this shape.)
        loader_nolabel = DataLoader(data_gen, batch_size=4, shuffle=False)
        batch_nolabel = next(iter(loader_nolabel))
        self.assertIsInstance(batch_nolabel, list)
        self.assertEqual(len(batch_nolabel), 1)
        self.assertIsInstance(batch_nolabel[0], torch.Tensor)
        self.assertEqual(batch_nolabel[0].shape[0], 4)

    # -------------------------------------------------#
    #        Application Functionality for 2D         #
    # -------------------------------------------------#
    # Usage: Grayscale without Labels
    def test_RUN_2D_GRAYSCALE_noLabel(self):
        data_gen = DataGenerator(
            self.sampleList_gray_2D, self.tmp_data.name, grayscale=True
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 1)
            # channel-first: (C, H, W)
            self.assertTrue(np.array_equal(sample[0].shape, (1, 224, 224)))

    # Usage: RGB without Labels
    def test_RUN_2D_RGB_noLabel(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D, self.tmp_data.name, grayscale=False
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 1)
            # channel-first: (C, H, W)
            self.assertTrue(np.array_equal(sample[0].shape, (3, 224, 224)))

    # Usage: With Labels
    def test_RUN_2D_withLabel(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            grayscale=False,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 2)
            # label is torch.tensor
            label = sample[1]
            self.assertTrue(np.array_equal(label.shape.numel(), (4)))

    # -------------------------------------------------#
    #        Application Functionality for 3D         #
    # -------------------------------------------------#
    # Usage: Grayscale without Labels
    def test_RUN_3D_GRAYSCALE_noLabel(self):
        data_gen = DataGenerator(
            self.sampleList_gray_3D,
            self.tmp_data.name,
            grayscale=True,
            two_dim=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 1)
            # channel-first: (C, D, H, W)
            self.assertTrue(np.array_equal(sample[0].shape, (1, 16, 16, 16)))

    # Usage: RGB without Labels
    def test_RUN_3D_RGB_noLabel(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_3D,
            self.tmp_data.name,
            grayscale=False,
            two_dim=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 1)
            # channel-first: (C, D, H, W)
            self.assertTrue(np.array_equal(sample[0].shape, (3, 16, 16, 16)))

    # Usage: With Labels
    def test_RUN_3D_withLabel(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_3D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            two_dim=False,
            grayscale=False,
            loader=numpy_loader,
            resize=None,
            standardize_mode=None,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 2)
            # label is torch.tensor
            label = sample[1]
            self.assertTrue(np.array_equal(label.shape.numel(), (4)))

    # -------------------------------------------------#
    #     Application Functionality with Metadata     #
    # -------------------------------------------------#
    # Usage: Metadata for inference
    def test_RUN_Metadata_noLabel(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            metadata=self.metadata,
            grayscale=False,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 1)
            self.assertTrue(len(sample[0]) == 2)
            # channel-first: (C, H, W)
            self.assertTrue(np.array_equal(sample[0][0].shape, (3, 224, 224)))
            # metadata is torch.tensor
            metadata = sample[0][1]
            self.assertTrue(np.array_equal(metadata.shape.numel(), (10)))

    # Usage: Metadata for training
    def test_RUN_Metadata_withLabel(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            metadata=self.metadata,
            grayscale=False,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 2)
            # label is torch.tensor
            label = sample[1]
            self.assertTrue(np.array_equal(label.shape.numel(), (4)))
            self.assertTrue(len(sample[0]) == 2)
            # channel-first: (C, H, W)
            self.assertTrue(np.array_equal(sample[0][0].shape, (3, 224, 224)))
            # metadata is torch.tensor
            metadata = sample[0][1]
            self.assertTrue(np.array_equal(metadata.shape.numel(), (10)))

    # -------------------------------------------------#
    #                 Multi-Processing                #
    # -------------------------------------------------#
    def test_MP(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            grayscale=False,
        )
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 2)
            self.assertTrue(np.array_equal(sample[1].shape.numel(), (4)))

    # -------------------------------------------------#
    #             Beforehand Preprocessing            #
    # -------------------------------------------------#
    def test_PrepareImages(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            prepare_images=True,
            grayscale=False,
        )
        precprocessed_images = os.listdir(data_gen.prepare_dir)
        self.assertTrue(len(precprocessed_images), len(self.sampleList_rgb_2D))
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 2)
            self.assertTrue(np.array_equal(sample[1].shape.numel(), (4)))
        shutil.rmtree(data_gen.prepare_dir)

    def test_PrepareImages_MP(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            prepare_images=True,
            grayscale=False,
        )
        precprocessed_images = os.listdir(data_gen.prepare_dir)
        self.assertTrue(len(precprocessed_images), len(self.sampleList_rgb_2D))
        for i in range(0, 10):
            sample = data_gen[i]
            self.assertTrue(len(sample), 2)
            self.assertTrue(np.array_equal(sample[1].shape.numel(), (4)))
        shutil.rmtree(data_gen.prepare_dir)

    # -------------------------------------------------#
    #      Integration: DataLoader + NeuralNetwork    #
    # -------------------------------------------------#
    # End-to-end check that a plain torch DataLoader wrapping a DataGenerator
    # (no manual has_labels/has_metadata/has_sample_weights on the loader
    # itself, unlike WrapperLoader) works with both train() and predict() --
    # this is the exact setup pipeline_torch_dummy.py uses.
    def _integration_model(self, resolution=(16, 16), n_labels=4, channels=3):
        return NeuralNetwork(
            n_labels=n_labels,
            channels=channels,
            architecture="2D.Vanilla",
            input_resolution=resolution,
            pretrained_weights=False,
        )

    def test_INTEGRATION_train_with_dataloader(self):
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(16, 16),
            standardize_mode="z-score",
            grayscale=False,
        )
        loader = DataLoader(data_gen, batch_size=5, shuffle=True)

        model = self._integration_model()
        hist = model.train(training_generator=loader, epochs=2)
        self.assertIn("loss", hist)
        self.assertEqual(len(hist["loss"]), 2)

    def test_INTEGRATION_predict_with_dataloader_noLabel(self):
        # Regression test: predict() must correctly infer has_labels=False
        # from loader.dataset (not the DataLoader instance) and correctly
        # unwrap the single-element batch that default_collate produces.
        data_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=None,
            resize=(16, 16),
            standardize_mode="z-score",
            grayscale=False,
        )
        loader = DataLoader(data_gen, batch_size=5, shuffle=False)

        model = self._integration_model()
        preds = model.predict(loader)
        self.assertEqual(preds.shape, (len(self.sampleList_rgb_2D), 4))
        for row in preds:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=4)

    def test_INTEGRATION_train_then_predict_roundtrip(self):
        train_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(16, 16),
            standardize_mode="z-score",
            grayscale=False,
        )
        train_loader = DataLoader(train_gen, batch_size=5, shuffle=True)

        test_gen = DataGenerator(
            self.sampleList_rgb_2D,
            self.tmp_data.name,
            labels=None,
            resize=(16, 16),
            standardize_mode="z-score",
            grayscale=False,
        )
        test_loader = DataLoader(test_gen, batch_size=5, shuffle=False)

        model = self._integration_model()
        model.train(training_generator=train_loader, epochs=2)
        preds = model.predict(test_loader)
        self.assertEqual(preds.shape, (len(self.sampleList_rgb_2D), 4))

    # -------------------------------------------------#
    #                   Utilization                   #
    # -------------------------------------------------#
    # Class Creation
    # def test_utils_iter(self):
    #    data_gen = DataGenerator(self.sampleList_rgb_2D, self.tmp_data.name,)
    #    counter = 0
    #    for sample in data_gen:
    #        if counter < 3:
    #            self.assertTrue(np.array_equal(sample[0].shape, (8,224,224,3)))
    #        else:
    #            self.assertTrue(np.array_equal(sample[0].shape, (1,224,224,3)))
    #        counter += 1
    #    self.assertTrue(counter == 4)
