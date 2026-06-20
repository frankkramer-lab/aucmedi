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
import csv
import logging


# -----------------------------------------------------#
#                   Custom Callbacks                  #
# -----------------------------------------------------#
class Callback:
    """Base class for custom callbacks that can be passed to NeuralNetwork.train().

    Custom callbacks should inherit from this class and implement the on_epoch_end() method.

    The on_epoch_end() method is called at the end of each epoch during training and receives the current epoch number, a logs dictionary containing the latest values of all logged metrics, and the model instance. It can perform any desired actions based on this information, such as logging, saving checkpoints, or implementing early stopping logic.

    Example usage:
    ```python
    class MyCustomCallback(Callback):
        def on_epoch_end(self, epoch=None, logs=None, model=None):
            # Custom logic here
            print(f"Epoch {epoch} ended with logs: {logs}")
    ```

    This callback can then be passed to NeuralNetwork.train() in the callbacks list.
    """

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        pass


# -----------------------------------------------------#
#                   Early Stopping                     #
# -----------------------------------------------------#
class EarlyStoppingCallback(Callback):
    """Base class for custom early stopping callbacks that can be passed to NeuralNetwork.train()."""

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        """Checks if training should be stopped based on the provided logs and epoch number.

        This method should return True if training should be stopped, and False otherwise. The logic for determining when to stop training should be implemented in subclasses of EarlyStoppingCallback.

        Example usage:
        ```python
        class MyEarlyStopping(EarlyStoppingCallback):
            def on_epoch_end(self, epoch=None, logs=None, model=None):
                # Custom early stopping logic here
                if logs and logs.get('val_loss') and logs['val_loss'][-1] < 0.1:
                    return True  # Stop training if validation loss is below 0.1
                return False
        ```

        This early stopping callback can then be passed to NeuralNetwork.train() in the early_stopping_callback parameter.
        """
        return False


class EarlyStopping(EarlyStoppingCallback):
    """
    Custom early stopping callback that monitors a specified metric and stops training if it doesn't improve for a given number of epochs (patience).
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param monitor: Metric within training log to be monitored (e.g., 'val_loss').
    :return: True if training should be stopped, False otherwise.
    """

    def __init__(self, patience=3, monitor="val_loss"):
        self.patience = patience
        self.monitor = monitor
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        # Safely get the latest value of the monitored metric
        if logs is None:
            logging.debug("EarlyStopping: logs is None on epoch %s.", epoch)
            return False
        if self.monitor not in logs:
            logging.debug(
                "EarlyStopping: monitor '%s' not in logs on epoch %s.",
                self.monitor,
                epoch,
            )
            return False
        vals = logs[self.monitor]
        if isinstance(vals, (list, tuple)):
            if len(vals) == 0:
                logging.debug(
                    "EarlyStopping: empty list for '%s' on epoch %s.",
                    self.monitor,
                    epoch,
                )
                return False
            current_loss = vals[-1]
        else:
            current_loss = vals

        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered on epoch %d.", epoch)
                return True
        return False


class ThresholdEarlyStopping(EarlyStoppingCallback):
    """Custom early stopping callback that monitors a specified metric and stops training if it doesn't improve for a given number of epochs (patience).
    The number of patience epochs are only counted when baseline loss is achieved.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param baseline: Baseline value for the monitored metric. Patience counting starts only after this baseline is attained.
    :param monitor: Metric within training log to be monitored (e.g., 'val_loss').
    :return: True if training should be stopped, False otherwise.
    """

    def __init__(self, patience=3, baseline=0.0, monitor="val_loss"):
        self.patience = patience
        self.monitor = monitor
        self.counter = 0
        self.best_loss = None
        self.baseline = baseline
        self.baseline_attained = False

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        # Safely get the latest value of the monitored metric
        if logs is None:
            logging.debug("ThresholdEarlyStopping: logs is None on epoch %s.", epoch)
            return False
        if self.monitor not in logs:
            logging.debug(
                "ThresholdEarlyStopping: monitor '%s' not in logs on epoch %s.",
                self.monitor,
                epoch,
            )
            return False
        vals = logs[self.monitor]
        if isinstance(vals, (list, tuple)):
            if len(vals) == 0:
                logging.debug(
                    "ThresholdEarlyStopping: empty list for '%s' on epoch %s.",
                    self.monitor,
                    epoch,
                )
                return False
            current_loss = vals[-1]
        else:
            current_loss = vals

        if not self.baseline_attained:
            if current_loss <= self.baseline:
                logging.info("Baseline attained at epoch %d.", epoch)
                self.baseline_attained = True
            else:
                return False  # Don't start counting patience until baseline is attained

        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered on epoch %d.", epoch)
                return True
        return False


class MinEpochEarlyStopping(EarlyStoppingCallback):
    """Custom early stopping callback that monitors a specified metric and stops training if it doesn't improve for a given number of epochs (patience).
    The number of patience epochs are only counted after a specified minimum epoch is reached.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param min_epoch: Minimum epoch after which patience counting starts.
    :param monitor: Metric within training log to be monitored (e.g., 'val_loss').
    :return: True if training should be stopped, False otherwise.
    """

    def __init__(self, patience=3, min_epoch=5, monitor="val_loss"):
        self.patience = patience
        self.monitor = monitor
        self.counter = 0
        self.best_loss = None
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        if epoch < self.min_epoch:
            return False  # Don't start counting patience until minimum epoch is reached
        # Safely get the latest value of the monitored metric
        if logs is None:
            logging.debug("MinEpochEarlyStopping: logs is None on epoch %s.", epoch)
            return False
        if self.monitor not in logs:
            logging.debug(
                "MinEpochEarlyStopping: monitor '%s' not in logs on epoch %s.",
                self.monitor,
                epoch,
            )
            return False
        vals = logs[self.monitor]
        if isinstance(vals, (list, tuple)):
            if len(vals) == 0:
                logging.debug(
                    "MinEpochEarlyStopping: empty list for '%s' on epoch %s.",
                    self.monitor,
                    epoch,
                )
                return False
            current_loss = vals[-1]
        else:
            current_loss = vals

        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered on epoch %d.", epoch)
                return True
        return False


# -----------------------------------------------------#
#                   Persistence                        #
# -----------------------------------------------------#
class ModelCheckpoint(Callback):
    """PyTorch replacement for tf.keras.callbacks.ModelCheckpoint(save_best_only=True).

    Stores the model weights via NeuralNetwork.dump() whenever the monitored
    metric improves.

    :param filepath: Path to store the best model weights (".pt"/".pth" recommended).
    :param monitor: Metric within training log to be monitored (e.g., 'val_loss').
    :param mode: Either "min" or "max", depending on whether the monitored metric
        should be minimized or maximized.
    """

    def __init__(self, filepath, monitor="val_loss", mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        if logs is None or not logs.get(self.monitor):
            logging.debug(
                "ModelCheckpoint: monitor '%s' not available on epoch %s.",
                self.monitor,
                epoch,
            )
            return None
        current = logs[self.monitor][-1]
        improved = self.best is None or (
            current < self.best if self.mode == "min" else current > self.best
        )
        if improved:
            self.best = current
            model.dump(self.filepath)
            logging.info(
                "ModelCheckpoint: saved model with %s=%.6f to %s on epoch %d.",
                self.monitor,
                current,
                self.filepath,
                epoch,
            )
        return None


class CSVLogger(Callback):
    """PyTorch replacement for tf.keras.callbacks.CSVLogger.

    Appends a row with the latest values of every logged metric after each epoch.

    :param filename: Path to the CSV file the logs are written to.
    :param separator: Delimiter used in the CSV file.
    :param append: If True, append to an existing file; otherwise overwrite it.
    """

    def __init__(self, filename, separator=",", append=True):
        self.filename = filename
        self.separator = separator
        self.append = append
        self._header_written = False

    def on_epoch_end(self, epoch=None, logs=None, model=None):
        if logs is None:
            return None
        row = {k: v[-1] for k, v in logs.items() if v}
        write_header = not self._header_written
        mode = "a" if (self.append or self._header_written) else "w"
        with open(self.filename, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter=self.separator)
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
        return None
