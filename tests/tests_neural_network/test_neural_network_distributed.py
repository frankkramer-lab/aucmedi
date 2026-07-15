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
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

# Internal libraries
from aucmedi import create_batch_loader, create_distributed_loader
from aucmedi.neural_network.model_distributed import NeuralNetwork
from aucmedi.utils.callbacks import MinEpochEarlyStopping


# -----------------------------------------------------#
#   Unittest: NeuralNetwork.train_distributed() guards #
# -----------------------------------------------------#
# These don't touch real CUDA/multiprocessing, so they run in any environment
# (including CI machines without a GPU) unlike the rest of this module.
class NeuralNetworkDistributedGuardsTEST(unittest.TestCase):
    def test_requires_cuda(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                model.train_distributed(generator_fn=lambda rank, world_size: None)

    def test_requires_world_size_at_least_two(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=1
        ):
            with self.assertRaises(ValueError):
                model.train_distributed(
                    generator_fn=lambda rank, world_size: None, world_size=1
                )


# -----------------------------------------------------#
#         Unittest: create_distributed_loader()        #
# -----------------------------------------------------#
class CreateDistributedLoaderTEST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234)
        cls.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )
        cls.sampleList_rgb = []
        for i in range(0, 11):  # odd count: exercises uneven sharding too
            img_rgb = np.random.rand(32, 32, 3) * 255
            pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = f"image.sample_{i}.RGB.png"
            pillow.save(os.path.join(cls.tmp_data.name, index))
            cls.sampleList_rgb.append(index)

        cls.labels_ohe = np.zeros((11, 4), dtype=np.uint8)
        for i in range(0, 11):
            cls.labels_ohe[i][np.random.randint(0, 4)] = 1

    def test_returns_callable(self):
        loader_fn = create_distributed_loader(
            self.sampleList_rgb, self.tmp_data.name, labels=self.labels_ohe,
            resize=(32, 32), batch_size=3, num_workers=0,
        )
        self.assertTrue(callable(loader_fn))

    def test_shard_is_wrapper_loader(self):
        loader_fn = create_distributed_loader(
            self.sampleList_rgb, self.tmp_data.name, labels=self.labels_ohe,
            resize=(32, 32), batch_size=3, num_workers=0,
        )
        loader = loader_fn(0, 2)
        self.assertTrue(loader.has_labels)

    def test_shards_disjoint_and_complete(self):
        world_size = 3
        loader_fn = create_distributed_loader(
            self.sampleList_rgb, self.tmp_data.name, labels=self.labels_ohe,
            resize=(32, 32), batch_size=3, num_workers=0,
        )
        seen = set()
        for rank in range(world_size):
            loader = loader_fn(rank, world_size)
            shard_samples = list(loader.samples)
            self.assertTrue(seen.isdisjoint(shard_samples))
            seen.update(shard_samples)
        self.assertEqual(seen, set(self.sampleList_rgb))

    def test_shard_labels_match_shard_samples(self):
        # samples[rank::world_size] and labels[rank::world_size] must line up:
        # each rank's shard has to keep (sample, label) pairs intact.
        world_size = 2
        loader_fn = create_distributed_loader(
            self.sampleList_rgb, self.tmp_data.name, labels=self.labels_ohe,
            resize=(32, 32), batch_size=64, num_workers=0,
        )
        for rank in range(world_size):
            loader = loader_fn(rank, world_size)
            expected_samples = self.sampleList_rgb[rank::world_size]
            expected_labels = self.labels_ohe[rank::world_size]
            self.assertEqual(list(loader.samples), expected_samples)
            self.assertTrue(np.array_equal(loader.labels, expected_labels))


# -----------------------------------------------------#
#      Unittest: NeuralNetwork.train_distributed()      #
# -----------------------------------------------------#
@unittest.skipUnless(
    torch.cuda.is_available(), "train_distributed() requires at least one CUDA GPU"
)
class NeuralNetworkDistributedTEST(unittest.TestCase):
    # Create random imaging and classification data
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234)
        cls.tmp_data = tempfile.TemporaryDirectory(
            prefix="tmp.aucmedi.", suffix=".data"
        )

        cls.sampleList_rgb = []
        for i in range(0, 12):
            img_rgb = np.random.rand(32, 32, 3) * 255
            pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = f"image.sample_{i}.RGB.png"
            pillow.save(os.path.join(cls.tmp_data.name, index))
            cls.sampleList_rgb.append(index)

        cls.labels_ohe = np.zeros((12, 4), dtype=np.uint8)
        for i in range(0, 12):
            cls.labels_ohe[i][np.random.randint(0, 4)] = 1

        # world_size=2: on a single-GPU machine this simulates 2 ranks sharing
        # the GPU over the "gloo" backend (see NeuralNetwork.train_distributed());
        # on a real multi-GPU machine each rank gets its own device.
        cls.world_size = 2

    def _make_loader_fn(self, shuffle=False):
        return create_distributed_loader(
            self.sampleList_rgb,
            self.tmp_data.name,
            labels=self.labels_ohe,
            resize=(32, 32),
            shuffle=shuffle,
            batch_size=3,
            num_workers=0,
        )

    # -------------------------------------------------#
    #                  Model Training                 #
    # -------------------------------------------------#
    def test_training_pure(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        hist = model.train_distributed(
            generator_fn=self._make_loader_fn(),
            world_size=self.world_size,
            epochs=2,
        )
        self.assertTrue("loss" in hist)
        self.assertEqual(len(hist["loss"]), 2)

    def test_training_validation(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        hist = model.train_distributed(
            generator_fn=self._make_loader_fn(),
            validation_generator_fn=self._make_loader_fn(),
            world_size=self.world_size,
            epochs=2,
        )
        self.assertTrue("loss" in hist and "val_loss" in hist)
        self.assertEqual(len(hist["loss"]), 2)
        self.assertEqual(len(hist["val_loss"]), 2)

    def test_training_transferlearning(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        hist = model.train_distributed(
            generator_fn=self._make_loader_fn(),
            validation_generator_fn=self._make_loader_fn(),
            world_size=self.world_size,
            epochs=2,
            transfer_learning=True,
            transfer_epochs=1,
        )
        self.assertTrue("tl_loss" in hist and "tl_val_loss" in hist)
        self.assertTrue("ft_loss" in hist and "ft_val_loss" in hist)

    def test_training_early_stopping_does_not_deadlock(self):
        # Regression test for the rank-0-gated early-stopping broadcast in
        # NeuralNetwork._train_epoch(): if the stop decision weren't broadcast
        # to every rank, a rank that stopped alone would hang the others on
        # the next collective all_reduce/broadcast call instead of returning.
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        early_stopping = MinEpochEarlyStopping(patience=1, monitor="val_loss", min_epoch=0)
        hist = model.train_distributed(
            generator_fn=self._make_loader_fn(),
            validation_generator_fn=self._make_loader_fn(),
            world_size=self.world_size,
            epochs=5,
            callbacks=[early_stopping],
        )
        self.assertTrue("loss" in hist and "val_loss" in hist)
        self.assertLessEqual(len(hist["loss"]), 5)

    # -------------------------------------------------#
    #                 Model Inference                 #
    # -------------------------------------------------#
    def test_predict_after_train_distributed(self):
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        model.train_distributed(
            generator_fn=self._make_loader_fn(),
            world_size=self.world_size,
            epochs=1,
        )
        # After train_distributed() returns, the parent process's model must be
        # a plain module again (not left wrapped in DistributedDataParallel),
        # since only the per-rank worker copies were ever DDP-wrapped.
        self.assertNotIsInstance(model.model, torch.nn.parallel.DistributedDataParallel)

        test_loader = create_batch_loader(
            self.sampleList_rgb, self.tmp_data.name, resize=(32, 32), batch_size=4,
            num_workers=0,
        )
        preds = model.predict(test_loader)
        self.assertEqual(preds.shape, (12, 4))
        for i in range(0, 12):
            self.assertTrue(np.sum(preds[i]) >= 0.99 and np.sum(preds[i]) <= 1.01)

    # -------------------------------------------------#
    #        Non-distributed train() still works       #
    # -------------------------------------------------#
    def test_train_still_works(self):
        # Regression test: model_distributed.NeuralNetwork.train() must remain
        # unaffected by the train_distributed()/_fit() refactor.
        model = NeuralNetwork(n_labels=4, channels=3, input_resolution=(32, 32))
        loader = create_batch_loader(
            self.sampleList_rgb, self.tmp_data.name, labels=self.labels_ohe,
            resize=(32, 32), batch_size=3, num_workers=0,
        )
        hist = model.train(training_generator=loader, epochs=2)
        self.assertTrue("loss" in hist)
        self.assertEqual(len(hist["loss"]), 2)


if __name__ == "__main__":
    unittest.main()
