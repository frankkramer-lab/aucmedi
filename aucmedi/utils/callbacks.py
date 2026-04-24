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
import logging


# -----------------------------------------------------#
#                   Custom Callbacks                  #
# -----------------------------------------------------#\
class ReduceLROnPlateau:
    """
    Custom learning rate scheduler that reduces the learning rate by a factor if a specified metric does not improve for a given number of epochs (patience).
    :param patience: Number of epochs with no improvement after which learning rate will be reduced.
    :param factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
    :param monitor: Metric within training log to be monitored (e.g., 'val_loss').
    :return: New learning rate if it was reduced, otherwise None.
    """

    def __init__(self, patience=3, factor=0.1, monitor="val_loss"):
        self.patience = patience
        self.factor = factor
        self.monitor = monitor
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, epoch=None, logs=None, lr=None):
        # Get the latest value of the monitored metric
        current_loss = logs[self.monitor][-1]
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = lr * self.factor  # Reduce learning rate by factor
                logging.info(
                    "Learning rate reduced to %.6f on epoch %d.", new_lr, epoch
                )
                return new_lr
        return None


class EarlyStopping:
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

    def on_epoch_end(self, epoch=None, logs=None):
        # Get the latest value of the monitored metric
        current_loss = logs[self.monitor][-1]
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered on epoch %d.", epoch)
                return True
        return False


class ThresholdEarlyStopping:
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

    def on_epoch_end(self, epoch=None, logs=None):
        current_loss = logs[self.monitor][-1]  # Get the latest
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


class MinEpochEarlyStopping:
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

    def on_epoch_end(self, epoch=None, logs=None):
        if epoch < self.min_epoch:
            return False  # Don't start counting patience until minimum epoch is reached

        current_loss = logs[self.monitor][-1]  # Get the latest
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered on epoch %d.", epoch)
                return True
        return False
