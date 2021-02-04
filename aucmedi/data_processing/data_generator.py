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
# External libraries
from tensorflow.keras.preprocessing.image import Iterator
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import repeat
import tempfile
import pickle
import os
# Internal libraries
from aucmedi.data_processing.io_data import image_loader
from aucmedi.data_processing.subfunctions import Standardize, Resize

#-----------------------------------------------------#
#                   Static Variables                  #
#-----------------------------------------------------#
ACCEPTABLE_IMAGE_FORMATS = ["jpeg", "jpg", "tif", "tiff", "png", "bmp", "gif"]

#-----------------------------------------------------#
#                 Keras Data Generator                #
#-----------------------------------------------------#
""" Infinite Data Generator which automatically creates batches from a list of samples.
    The created batches are model ready. This generator can be supplied directly
    to the keras model fit() function.

    The Data Generator can be used for training, validation as well as for prediction.
    It supports real-time batch generation as well as beforehand preparation of batches,
    which are then temporarly stored on disk.

    The resulting batches are created based the following pipeline:
    - Image Loading
    - Optional application of Data Augmentation
    - Optional application of Subfunctions
    - Standardize image
    - Stacking processed images to a batch

    Build on top of Keras Iterator:
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/Iterator
"""
class DataGenerator(Iterator):
    #-----------------------------------------------------#
    #                    Initialization                   #
    #-----------------------------------------------------#
    """Initialization function of the Data Generator which acts as a configuraiton hub.

        If using for prediction, the 'labels' parameter have to be None.
        Data augmentation is applied even for prediction if a DataAugmentation object is provided!
        Applying 'None' to resize will result into no image resizing. Default (224, 224)

        Arguments:
            samples (List of Strings):      List of sample/index encoded as Strings.
            path_imagedir (String):         Path to the directory containing the images.
            labels (NumPy Array):           Classification list with One-Hot Encoding.
            image_format (String):          Image format to add at the end of the sample index for image loading.
            batch_size (Integer):           Number of samples inside a single batch.
            resize (Tuple of Integers):     Resizing shape consisting of a X and Y size.
            img_aug (ImageAugmentation):    Image Augmentation class instance which performs diverse data augmentation techniques.
            shuffle (Boolean):              Boolean, whether dataset should be shuffled.
            grayscale (Boolean):            Boolean, whether images are grayscale or RGB.
            subfunctions (List of Subfunctions):
                                            List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
            standardize_mode (String):      Standardization modus in which image intensity values are scaled.
            prepare_images (Boolean):       Boolean, whether all images should be prepared and backup to disk before training.
            sample_weights (List of Floats):List of weights for samples.
            seed (Integer):                 Seed to ensure reproducibility for random function.
            workers (Integer):              Number of workers. If n_workers > 1 = use multi-threading for image preprocessing.
    """
    def __init__(self, samples, path_imagedir, labels=None, image_format=None,
                 batch_size=32, resize=(224, 224), img_aug=None, shuffle=False,
                 grayscale=False, subfunctions=[], standardize_mode="tf",
                 prepare_images=False, sample_weights=None, seed=None,
                 workers=1):
        # Cache class variables
        self.labels = labels
        self.sample_weights = sample_weights
        self.prepare_images = prepare_images
        self.workers = workers
        self.samples = samples
        self.path_imagedir = path_imagedir
        self.image_format = image_format
        self.grayscale = grayscale
        self.subfunctions = subfunctions
        self.img_aug = img_aug
        # Initialize Standardization Subfunction
        if standardize_mode is not None:
            self.sf_standardize = Standardize(mode=standardize_mode)
        else : self.sf_standardize = None
        # Initialize Resizing Subfunction
        if resize is not None : self.sf_resize = Resize(shape=resize)
        else : self.sf_resize = None
        # Sanity check for label correctness
        if labels is not None and len(samples) != len(labels):
            raise ValueError("Samples and labels do not have same size!",
                             len(samples), len(labels))
        # Sanity check for sample weights correctness
        if sample_weights is not None and len(samples) != len(sample_weights):
            raise ValueError("Samples and sample weights do not have same size!",
                             len(samples), len(sample_weights))
        # Verify that labels and sample weights are NumPy arrays
        if labels is not None and not isinstance(labels, np.ndarray):
            self.labels = np.asarray(self.labels)
        if sample_weights is not None and not isinstance(sample_weights,
                                                         np.ndarray):
            self.sample_weights = np.asarray(self.sample_weights)

        # If prepare_image modus activated
        # -> Preprocess images beforehand and store them to disk for fast usage later
        if self.prepare_images:
            tmp_dir = tempfile.mkdtemp(prefix="aucmedi.tmp.", suffix=".data")
            self.prepare_dir = tmp_dir
            for i in range(0, len(samples)):
                preproc_img = self.preprocess_image(index=i,
                                                    prepared_batch=False)
                path_img = os.path.join(tmp_dir, "img_" + str(i))
                with open(path_img + ".pickle", "wb") as pickle_writer:
                    pickle.dump(preproc_img, pickle_writer)
            print("A directory for image preparation was created:", tmp_dir)

        # Pass initialization parameters to parent Iterator class
        size = len(samples)
        super(DataGenerator, self).__init__(size, batch_size, shuffle, seed)

    #-----------------------------------------------------#
    #              Batch Generation Function              #
    #-----------------------------------------------------#
    """Function for batch generation given a list of random selected samples."""
    def _get_batches_of_transformed_samples(self, index_array):
        # Initialize Batch stack
        batch_stack = ([],)
        if self.labels is not None : batch_stack += ([],)
        if self.sample_weights is not None : batch_stack += ([],)

        # Process image for each index - Sequential
        if self.workers == 0 or self.workers == 1:
            for i in index_array:
                batch_img = self.preprocess_image(index=i,
                                                  prepared_batch=self.prepare_images)
                batch_stack[0].append(batch_img)
        # Process image for each index - Multi-threading
        else:
            with ThreadPool(self.workers) as pool:
                mp_params = zip(index_array, repeat(self.prepare_images))
                batches_img = pool.starmap(self.preprocess_image, mp_params)
            batch_stack[0].extend(batches_img)

        # Add classification to batch if available
        if self.labels is not None:
            batch_stack[1].extend(self.labels[index_array])
        # Add sample weight to batch if available
        if self.sample_weights is not None:
            batch_stack[2].extend(self.sample_weights[index_array])

        # Stack images together into a batch
        batch = (np.stack(batch_stack[0], axis=0), )
        # Stack classifications together into a batch if available
        if self.labels is not None:
            batch += (np.stack(batch_stack[1], axis=0), )
        # Stack sample weights together into a batch if available
        if self.sample_weights is not None:
            batch += (np.stack(batch_stack[2], axis=0), )
        # Return generated Batch
        return batch

    #-----------------------------------------------------#
    #                 Image Preprocessing                 #
    #-----------------------------------------------------#
    """Preprocessing function for applying subfunctions, augmentation, resizing and standardization
       on an image given its index.

       Activating the prepared_batch option also allows loading a beforehand preprocessed image from disk.
    """
    def preprocess_image(self, index, prepared_batch=False):
        # Load prepared image from disk
        if prepared_batch:
            path_img = os.path.join(self.prepare_dir, "img_" + str(index))
            with open(path_img + ".pickle", "rb") as pickle_loader:
                img = pickle.load(pickle_loader)
        # Preprocess image during runtime
        else:
            # Load image
            img = image_loader(self.samples[index], self.path_imagedir,
                               image_format=self.image_format,
                               grayscale=self.grayscale)
            # Apply image augmentation on image if activated
            if self.img_aug is not None:
                img = self.img_aug.apply(img)
            # Apply subfunctions on image
            for sf in self.subfunctions:
                img = sf.transform(img)
            # Apply resizing on image if activated
            if self.sf_resize is not None:
                img = self.sf_resize.transform(img)
            # Apply standardization on image if activated
            if self.sf_standardize is not None:
                img = self.sf_standardize.transform(img)
        # Return preprocessed image
        return img
