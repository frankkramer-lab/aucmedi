# ==============================================================================#
#  Author:       Fabian Wehr                                                   #
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
import math
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from aucmedi import NeuralNetwork
from aucmedi.data_processing.subfunctions import Standardize


# ==============================================================================#
#  Minimal generic DataLoader implementation under test                         #
# ==============================================================================#
class _ImageDataset(Dataset):
    """Sample-level Dataset: __getitem__ returns one (image, label) pair.

    Exposes has_labels / has_metadata / has_sample_weights so that
    NeuralNetwork.train() / predict() can inspect the loader contract via
    getattr(loader.dataset, "has_labels", True) etc.
    """

    def __init__(self, samples, path_imagedir, labels, img_size, standardize_mode):
        self.samples = samples
        self.path_imagedir = path_imagedir
        self.labels = labels
        self.img_size = img_size  # (H, W)
        self._sf_std = Standardize(mode=standardize_mode) if standardize_mode else None
        self.has_labels = labels is not None
        self.has_metadata = False
        self.has_sample_weights = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = (
            Image.open(os.path.join(self.path_imagedir, self.samples[idx]))
            .convert("RGB")
            .resize((self.img_size[1], self.img_size[0]))  # PIL: (W, H)
        )
        img = np.array(img, dtype=np.float32)           # (H, W, C)
        if self._sf_std is not None:
            img = self._sf_std.transform(img)
        img = np.transpose(img, (2, 0, 1))              # (C, H, W)
        img_t = torch.from_numpy(img)
        if self.labels is not None:
            return img_t, torch.from_numpy(self.labels[idx])
        return img_t


class _AucmediDataLoader(DataLoader):
    """DataLoader subclass.

    model.predict() reads has_* from the loader itself (not dataset), so they
    must be set on the DataLoader instance too.
    """

    def __init__(self, dataset: _ImageDataset, batch_size, shuffle, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.has_labels = dataset.has_labels
        self.has_metadata = dataset.has_metadata
        self.has_sample_weights = dataset.has_sample_weights


# ==============================================================================#
#  Test suite                                                                    #
# ==============================================================================#
class GenericDataLoaderTEST(unittest.TestCase):

    N_SAMPLES  = 20
    N_CLASSES  = 4
    IMG_SIZE   = (32, 32)
    BATCH_SIZE = 4
    STD_MODE   = "z-score"

    @classmethod
    def setUpClass(cls):
        np.random.seed(1234)
        cls.tmp = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.", suffix=".data")

        cls.samples = []
        for i in range(cls.N_SAMPLES):
            arr = (np.random.rand(*cls.IMG_SIZE, 3) * 255).astype(np.uint8)
            name = f"sample_{i:04d}.png"
            Image.fromarray(arr).save(os.path.join(cls.tmp.name, name))
            cls.samples.append(name)

        cls.labels = np.zeros((cls.N_SAMPLES, cls.N_CLASSES), dtype=np.float32)
        for i in range(cls.N_SAMPLES):
            cls.labels[i, i % cls.N_CLASSES] = 1.0

        cls.ds_labeled   = _ImageDataset(cls.samples, cls.tmp.name, cls.labels,   cls.IMG_SIZE, cls.STD_MODE)
        cls.ds_unlabeled = _ImageDataset(cls.samples, cls.tmp.name, None,          cls.IMG_SIZE, cls.STD_MODE)
        cls.loader       = _AucmediDataLoader(cls.ds_labeled,   cls.BATCH_SIZE, shuffle=False)
        cls.loader_nolbl = _AucmediDataLoader(cls.ds_unlabeled, cls.BATCH_SIZE, shuffle=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ------------------------------------------------------------------#
    #  GROUP: _ImageDataset contract                                     #
    # ------------------------------------------------------------------#
    def test_DATASET_attributes_with_labels(self):
        self.assertTrue(self.ds_labeled.has_labels)
        self.assertFalse(self.ds_labeled.has_metadata)
        self.assertFalse(self.ds_labeled.has_sample_weights)

    def test_DATASET_attributes_no_labels(self):
        self.assertFalse(self.ds_unlabeled.has_labels)
        self.assertFalse(self.ds_unlabeled.has_metadata)
        self.assertFalse(self.ds_unlabeled.has_sample_weights)

    def test_DATASET_len(self):
        self.assertEqual(len(self.ds_labeled), self.N_SAMPLES)
        self.assertEqual(len(self.ds_unlabeled), self.N_SAMPLES)

    def test_DATASET_item_shape_with_labels(self):
        img, lbl = self.ds_labeled[0]
        self.assertEqual(img.shape, (3, *self.IMG_SIZE))
        self.assertEqual(lbl.shape, (self.N_CLASSES,))

    def test_DATASET_item_dtype(self):
        img, lbl = self.ds_labeled[0]
        self.assertEqual(img.dtype, torch.float32)
        self.assertEqual(lbl.dtype, torch.float32)

    def test_DATASET_item_no_labels(self):
        item = self.ds_unlabeled[0]
        # Should be a plain tensor, not a tuple
        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.shape, (3, *self.IMG_SIZE))

    # ------------------------------------------------------------------#
    #  GROUP: _AucmediDataLoader contract                                #
    # ------------------------------------------------------------------#
    def test_LOADER_attributes_mirrored(self):
        self.assertEqual(self.loader.has_labels,        self.ds_labeled.has_labels)
        self.assertEqual(self.loader.has_metadata,      self.ds_labeled.has_metadata)
        self.assertEqual(self.loader.has_sample_weights, self.ds_labeled.has_sample_weights)

    def test_LOADER_len(self):
        expected = math.ceil(self.N_SAMPLES / self.BATCH_SIZE)
        self.assertEqual(len(self.loader), expected)

    def test_LOADER_batch_shape(self):
        imgs, lbls = next(iter(self.loader))
        self.assertEqual(imgs.shape[1:], (3, *self.IMG_SIZE))
        self.assertEqual(lbls.shape[1], self.N_CLASSES)
        self.assertLessEqual(imgs.shape[0], self.BATCH_SIZE)

    def test_LOADER_no_labels_batch_shape(self):
        batch = next(iter(self.loader_nolbl))
        # Without labels the DataLoader collates into a plain tensor
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape[1:], (3, *self.IMG_SIZE))

    def test_LOADER_covers_all_samples(self):
        total = sum(imgs.shape[0] for imgs, _ in self.loader)
        self.assertEqual(total, self.N_SAMPLES)

    # ------------------------------------------------------------------#
    #  GROUP: NeuralNetwork.train() integration                          #
    # ------------------------------------------------------------------#
    def _model(self):
        return NeuralNetwork(
            n_labels=self.N_CLASSES,
            channels=3,
            architecture="2D.Vanilla",
            input_resolution=self.IMG_SIZE,
            pretrained_weights=False,
        )

    def test_TRAIN_pure(self):
        hist = self._model().train(training_generator=self.loader, epochs=2)
        self.assertIn("loss", hist)
        self.assertEqual(len(hist["loss"]), 2)

    def test_TRAIN_with_validation(self):
        hist = self._model().train(
            training_generator=self.loader,
            validation_generator=self.loader,
            epochs=2,
        )
        self.assertIn("loss", hist)
        self.assertIn("val_loss", hist)
        self.assertEqual(len(hist["val_loss"]), 2)

    def test_TRAIN_transferlearning(self):
        hist = self._model().train(
            training_generator=self.loader,
            validation_generator=self.loader,
            epochs=2,
            transfer_learning=True,
        )
        self.assertIn("tl_loss", hist)
        self.assertIn("ft_loss", hist)
        self.assertIn("tl_val_loss", hist)
        self.assertIn("ft_val_loss", hist)

    def test_TRAIN_scheduler(self):
        hist = self._model().train(
            training_generator=self.loader,
            validation_generator=self.loader,
            epochs=2,
            scheduler=lr_scheduler.ReduceLROnPlateau,
        )
        self.assertIn("learning_rate", hist)

    # ------------------------------------------------------------------#
    #  GROUP: NeuralNetwork.predict() integration                        #
    # ------------------------------------------------------------------#
    def test_PREDICT_with_labels(self):
        preds = self._model().predict(self.loader)
        self.assertEqual(preds.shape, (self.N_SAMPLES, self.N_CLASSES))
        for row in preds:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)

    def test_PREDICT_no_labels(self):
        preds = self._model().predict(self.loader_nolbl)
        self.assertEqual(preds.shape, (self.N_SAMPLES, self.N_CLASSES))
        for row in preds:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)


# ==============================================================================#
if __name__ == "__main__":
    unittest.main()
