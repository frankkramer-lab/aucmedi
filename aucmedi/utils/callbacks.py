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
# -----------------------------------------------------#
class EarlyStopping:
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
