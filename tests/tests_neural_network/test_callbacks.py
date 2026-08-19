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
import csv
import os
import tempfile
import unittest

from aucmedi.utils.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    EarlyStoppingCallback,
    MinEpochEarlyStopping,
    ModelCheckpoint,
    ThresholdEarlyStopping,
)


# Minimal model stub: records dump() calls
class _MockModel:
    def __init__(self):
        self.dump_calls = []

    def dump(self, path):
        self.dump_calls.append(path)


def _logs(val_loss_seq):
    """Build a logs dict that mimics NeuralNetwork history format."""
    return {"val_loss": list(val_loss_seq)}


# -----------------------------------------------------#
#                  Unittest: Callbacks                #
# -----------------------------------------------------#
class CallbackTEST(unittest.TestCase):
    # -------------------------------------------------#
    #              Base Callback                      #
    # -------------------------------------------------#
    def test_BASE_create(self):
        cb = Callback()
        self.assertIsInstance(cb, Callback)

    def test_BASE_on_epoch_end_returns_none(self):
        cb = Callback()
        result = cb.on_epoch_end(epoch=0, logs={}, model=None)
        self.assertIsNone(result)

    # -------------------------------------------------#
    #           EarlyStoppingCallback base            #
    # -------------------------------------------------#
    def test_ESBase_is_callback(self):
        cb = EarlyStoppingCallback()
        self.assertIsInstance(cb, Callback)

    def test_ESBase_always_returns_false(self):
        cb = EarlyStoppingCallback()
        self.assertFalse(cb.on_epoch_end(epoch=0, logs=_logs([0.5]), model=None))
        self.assertFalse(cb.on_epoch_end(epoch=1, logs=None, model=None))

    # -------------------------------------------------#
    #                EarlyStopping                    #
    # -------------------------------------------------#
    def test_ES_create(self):
        cb = EarlyStopping(patience=3, monitor="val_loss")
        self.assertIsInstance(cb, EarlyStoppingCallback)
        self.assertEqual(cb.patience, 3)
        self.assertEqual(cb.monitor, "val_loss")
        self.assertIsNone(cb.best_loss)
        self.assertEqual(cb.counter, 0)

    def test_ES_none_logs_returns_false(self):
        cb = EarlyStopping()
        self.assertFalse(cb.on_epoch_end(epoch=0, logs=None))

    def test_ES_missing_monitor_returns_false(self):
        cb = EarlyStopping(monitor="val_loss")
        self.assertFalse(cb.on_epoch_end(epoch=0, logs={"loss": [0.5]}))

    def test_ES_empty_list_returns_false(self):
        cb = EarlyStopping()
        self.assertFalse(cb.on_epoch_end(epoch=0, logs={"val_loss": []}))

    def test_ES_scalar_value(self):
        # logs value as scalar (not list) should also work
        cb = EarlyStopping(patience=2)
        self.assertFalse(cb.on_epoch_end(epoch=0, logs={"val_loss": 0.5}))
        self.assertFalse(cb.on_epoch_end(epoch=1, logs={"val_loss": 0.6}))
        self.assertTrue(cb.on_epoch_end(epoch=2, logs={"val_loss": 0.7}))

    def test_ES_improves_each_epoch_never_stops(self):
        cb = EarlyStopping(patience=2)
        for epoch, loss in enumerate([0.9, 0.8, 0.7, 0.6, 0.5]):
            result = cb.on_epoch_end(epoch=epoch, logs=_logs([loss]))
            self.assertFalse(result)
        self.assertEqual(cb.counter, 0)

    def test_ES_no_improvement_triggers_after_patience(self):
        cb = EarlyStopping(patience=3)
        # First epoch: establishes best
        self.assertFalse(cb.on_epoch_end(epoch=0, logs=_logs([0.5])))
        # Three stagnant epochs → trigger on the third
        self.assertFalse(cb.on_epoch_end(epoch=1, logs=_logs([0.6])))
        self.assertFalse(cb.on_epoch_end(epoch=2, logs=_logs([0.6])))
        self.assertTrue(cb.on_epoch_end(epoch=3, logs=_logs([0.6])))

    def test_ES_counter_resets_on_improvement(self):
        cb = EarlyStopping(patience=3)
        cb.on_epoch_end(epoch=0, logs=_logs([0.5]))  # best=0.5
        cb.on_epoch_end(epoch=1, logs=_logs([0.6]))  # counter=1
        cb.on_epoch_end(epoch=2, logs=_logs([0.6]))  # counter=2
        cb.on_epoch_end(epoch=3, logs=_logs([0.4]))  # improvement → counter=0
        self.assertEqual(cb.counter, 0)
        self.assertAlmostEqual(cb.best_loss, 0.4)

    def test_ES_best_loss_tracks_minimum(self):
        cb = EarlyStopping()
        cb.on_epoch_end(epoch=0, logs=_logs([0.8]))
        self.assertAlmostEqual(cb.best_loss, 0.8)
        cb.on_epoch_end(epoch=1, logs=_logs([0.5]))
        self.assertAlmostEqual(cb.best_loss, 0.5)
        cb.on_epoch_end(epoch=2, logs=_logs([0.7]))  # no improvement
        self.assertAlmostEqual(cb.best_loss, 0.5)

    # -------------------------------------------------#
    #           ThresholdEarlyStopping                #
    # -------------------------------------------------#
    def test_TES_create(self):
        cb = ThresholdEarlyStopping(patience=2, baseline=0.4, monitor="val_loss")
        self.assertIsInstance(cb, EarlyStoppingCallback)
        self.assertFalse(cb.baseline_attained)

    def test_TES_returns_false_before_baseline(self):
        cb = ThresholdEarlyStopping(patience=2, baseline=0.4)
        # Loss above baseline — patience must not start
        for epoch in range(10):
            self.assertFalse(cb.on_epoch_end(epoch=epoch, logs=_logs([0.9])))
        self.assertFalse(cb.baseline_attained)
        self.assertEqual(cb.counter, 0)

    def test_TES_baseline_flag_flips(self):
        cb = ThresholdEarlyStopping(patience=2, baseline=0.5)
        cb.on_epoch_end(epoch=0, logs=_logs([0.3]))  # 0.3 <= 0.5 → attained
        self.assertTrue(cb.baseline_attained)

    def test_TES_stops_after_patience_post_baseline(self):
        cb = ThresholdEarlyStopping(patience=2, baseline=0.5)
        cb.on_epoch_end(epoch=0, logs=_logs([0.4]))  # attain baseline, best=0.4
        self.assertFalse(cb.on_epoch_end(epoch=1, logs=_logs([0.5])))  # counter=1
        self.assertTrue(cb.on_epoch_end(epoch=2, logs=_logs([0.5])))   # counter=2 → stop

    def test_TES_counter_resets_on_improvement_after_baseline(self):
        cb = ThresholdEarlyStopping(patience=3, baseline=0.5)
        cb.on_epoch_end(epoch=0, logs=_logs([0.4]))  # baseline attained, best=0.4
        cb.on_epoch_end(epoch=1, logs=_logs([0.5]))  # counter=1
        cb.on_epoch_end(epoch=2, logs=_logs([0.3]))  # improvement → counter=0
        self.assertEqual(cb.counter, 0)

    # -------------------------------------------------#
    #           MinEpochEarlyStopping                 #
    # -------------------------------------------------#
    def test_MES_create(self):
        cb = MinEpochEarlyStopping(patience=2, min_epoch=5)
        self.assertIsInstance(cb, EarlyStoppingCallback)
        self.assertEqual(cb.min_epoch, 5)

    def test_MES_returns_false_before_min_epoch(self):
        cb = MinEpochEarlyStopping(patience=1, min_epoch=5)
        # Even with stagnant loss, must not stop before epoch 5
        for epoch in range(5):
            self.assertFalse(cb.on_epoch_end(epoch=epoch, logs=_logs([0.9])))

    def test_MES_stops_after_patience_post_min_epoch(self):
        cb = MinEpochEarlyStopping(patience=2, min_epoch=3)
        for epoch in range(3):
            cb.on_epoch_end(epoch=epoch, logs=_logs([0.9]))  # ignored
        cb.on_epoch_end(epoch=3, logs=_logs([0.5]))          # best=0.5
        self.assertFalse(cb.on_epoch_end(epoch=4, logs=_logs([0.6])))  # counter=1
        self.assertTrue(cb.on_epoch_end(epoch=5, logs=_logs([0.6])))   # counter=2 → stop

    def test_MES_none_logs_after_min_epoch_returns_false(self):
        cb = MinEpochEarlyStopping(patience=2, min_epoch=2)
        self.assertFalse(cb.on_epoch_end(epoch=5, logs=None))

    # -------------------------------------------------#
    #               ModelCheckpoint                   #
    # -------------------------------------------------#
    def test_MC_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(os.path.join(tmp, "best.pt"), monitor="val_loss", mode="min")
            self.assertIsInstance(cb, Callback)
            self.assertIsNone(cb.best)

    def test_MC_none_logs_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(os.path.join(tmp, "best.pt"))
            result = cb.on_epoch_end(epoch=0, logs=None, model=_MockModel())
            self.assertIsNone(result)

    def test_MC_missing_monitor_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(os.path.join(tmp, "best.pt"), monitor="val_loss")
            result = cb.on_epoch_end(epoch=0, logs={"loss": [0.5]}, model=_MockModel())
            self.assertIsNone(result)

    def test_MC_saves_on_first_call_mode_min(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            cb = ModelCheckpoint(path, monitor="val_loss", mode="min")
            model = _MockModel()
            cb.on_epoch_end(epoch=0, logs=_logs([0.5]), model=model)
            self.assertEqual(model.dump_calls, [path])
            self.assertAlmostEqual(cb.best, 0.5)

    def test_MC_saves_on_improvement_min(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            cb = ModelCheckpoint(path, mode="min")
            model = _MockModel()
            cb.on_epoch_end(epoch=0, logs=_logs([0.5]), model=model)  # saved
            cb.on_epoch_end(epoch=1, logs=_logs([0.4]), model=model)  # saved (improved)
            cb.on_epoch_end(epoch=2, logs=_logs([0.6]), model=model)  # not saved
            self.assertEqual(len(model.dump_calls), 2)
            self.assertAlmostEqual(cb.best, 0.4)

    def test_MC_saves_on_improvement_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            cb = ModelCheckpoint(path, monitor="val_loss", mode="max")
            model = _MockModel()
            cb.on_epoch_end(epoch=0, logs=_logs([0.5]), model=model)  # saved
            cb.on_epoch_end(epoch=1, logs=_logs([0.3]), model=model)  # not saved (worse)
            cb.on_epoch_end(epoch=2, logs=_logs([0.8]), model=model)  # saved (improved)
            self.assertEqual(len(model.dump_calls), 2)
            self.assertAlmostEqual(cb.best, 0.8)

    def test_MC_does_not_save_without_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(os.path.join(tmp, "best.pt"), mode="min")
            model = _MockModel()
            cb.on_epoch_end(epoch=0, logs=_logs([0.5]), model=model)
            cb.on_epoch_end(epoch=1, logs=_logs([0.5]), model=model)  # equal → no save
            cb.on_epoch_end(epoch=2, logs=_logs([0.9]), model=model)  # worse → no save
            self.assertEqual(len(model.dump_calls), 1)

    # -------------------------------------------------#
    #                  CSVLogger                      #
    # -------------------------------------------------#
    def test_CSV_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = CSVLogger(os.path.join(tmp, "log.csv"))
            self.assertIsInstance(cb, Callback)
            self.assertFalse(cb._header_written)

    def test_CSV_none_logs_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = CSVLogger(os.path.join(tmp, "log.csv"))
            result = cb.on_epoch_end(epoch=0, logs=None, model=None)
            self.assertIsNone(result)

    def test_CSV_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            cb = CSVLogger(path)
            cb.on_epoch_end(epoch=0, logs={"val_loss": [0.5], "loss": [0.6]})
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertIn("val_loss", rows[0])
            self.assertIn("loss", rows[0])

    def test_CSV_appends_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            cb = CSVLogger(path, append=True)
            for epoch in range(3):
                cb.on_epoch_end(
                    epoch=epoch,
                    logs={"val_loss": [0.5 - epoch * 0.1], "loss": [0.6 - epoch * 0.1]},
                )
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 3)

    def test_CSV_overwrite_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            # First logger writes 2 epochs
            cb1 = CSVLogger(path, append=False)
            cb1.on_epoch_end(epoch=0, logs={"val_loss": [0.5]})
            cb1.on_epoch_end(epoch=1, logs={"val_loss": [0.4]})
            # New logger with append=False should overwrite → only 1 new row
            cb2 = CSVLogger(path, append=False)
            cb2.on_epoch_end(epoch=0, logs={"val_loss": [0.9]})
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["val_loss"]), 0.9)

    def test_CSV_custom_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            cb = CSVLogger(path, separator=";")
            cb.on_epoch_end(epoch=0, logs={"val_loss": [0.5], "loss": [0.6]})
            with open(path) as f:
                content = f.read()
            self.assertIn(";", content)

    def test_CSV_header_written_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            cb = CSVLogger(path)
            for epoch in range(5):
                cb.on_epoch_end(epoch=epoch, logs={"val_loss": [0.5]})
            with open(path) as f:
                lines = f.readlines()
            # 1 header line + 5 data lines
            self.assertEqual(len(lines), 6)
