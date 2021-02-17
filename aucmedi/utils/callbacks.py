#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from tensorflow.keras.callbacks import EarlyStopping

#-----------------------------------------------------#
#                   Custom Callbacks                  #
#-----------------------------------------------------#
"""
Author:   JBSnorro
Source:   https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0

Changed baseline to act as a real baseline.
The number of patience epochs are only counted when baseline loss is achieved.
"""
class ThresholdEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.baseline_attained = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.baseline_attained:
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.monitor_op(current, self.baseline):
                if self.verbose > 0:
                    print('Baseline attained.')
                self.baseline_attained = True
            else:
                return
        super(ThresholdEarlyStopping, self).on_epoch_end(epoch, logs)


"""
Author:   McLawrence
Source:   https://stackoverflow.com/questions/46287403/is-there-a-way-to-implement-early-stopping-in-keras-only-after-the-first-say-1

Changed baseline to act as a real baseline.
The number of patience epochs are only counted when baseline loss is achieved.
"""
class MinEpochEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
                 mode='auto', start_epoch = 100): # add argument for starting epoch
        super(MinEpochEarlyStopping, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super(MinEpochEarlyStopping, self).on_epoch_end(epoch, logs)
