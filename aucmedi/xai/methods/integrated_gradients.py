#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2023 IT-Infrastructure for Translational Medical Research,    #
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
# External Libraries
import numpy as np
import tensorflow as tf
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                 Integrated Gradients                #
#-----------------------------------------------------#
class IntegratedGradients(XAImethod_Base):
    """ XAI Method for Integrated Gradients.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Author: Aakash Kumar Nain <br>
        GitHub Profile: [https://github.com/AakashKumarNain](https://github.com/AakashKumarNain) <br>
        Date: Jun 02, 2020 <br>
        [https://keras.io/examples/vision/integrated_gradients](https://keras.io/examples/vision/integrated_gradients) <br>

    ??? abstract "Reference - Publication"
        Mukund Sundararajan, Ankur Taly, Qiqi Yan. 04 Mar 2017.
        Axiomatic Attribution for Deep Networks.
        <br>
        [https://arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365)

    This class provides functionality for running the compute_heatmap function,
    which computes an Integrated Gradients Map for an image with a model.
    """
    def __init__(self, model, layerName=None, num_steps=50):
        """ Initialization function for creating a Integrated Gradients Map as XAI Method object.

        Args:
            model (keras.model):            Keras model object.
            layerName (str):                Not required in Integrated Gradients Maps, but defined by Abstract Base Class.
            num_steps (int):                Number of iterations for interpolation.
        """
        # Cache class parameters
        self.model = model
        self.num_steps = num_steps
        #create baseline entry in constructor

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, eps=1e-8):
        """ Core function for computing the Integrated Gradients Map for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap potentially loses the channel axis.

        Returns:
            heatmap (numpy.ndarray):            Computed Integrated Gradients Map for provided image.
        """
        class_index = tf.convert_to_tensor(class_index, dtype=tf.int32)
        
        batch_size = len(image)
        
        # Perform interpolation
        hm = []
        baseline = np.zeros(image[0].shape).astype(np.float32) #TODO should not always be Zero. should be defined in constructor
        interpolated_imgs = np.zeros((self.num_steps + 1,) + baseline.shape) #memory is allocated once here and then reused
        for img in image:
            #the following two operations can also be considered a matrix multiplication if the data is arranhged differently. However the rearranging may be more expensive than this
            interpolated_imgs = np.einsum("B...,B->B...", 
                                        np.repeat(img[None, ...], self.num_steps + 1, axis = 0), 
                                        np.arange(self.num_steps + 1) / self.num_steps)
            interpolated_imgs += np.einsum("B...,B->B...",
                                        np.repeat(baseline[None, ...], self.num_steps + 1, axis = 0), 
                                        1 - (np.arange(self.num_steps + 1) / self.num_steps))

            # Get the gradients for each interpolated image
            grads = []
            for int_img in range(0, len(interpolated_imgs), batch_size): #batch gets evaluated n times
                int_imgs = tf.cast(interpolated_imgs[int_img:int_img + batch_size], tf.float32)
                # Compute gradient
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(int_imgs)
                    preds = self.model(int_imgs)
                    loss = tf.gather(preds, class_index, axis = 1)
                gradient = tape.gradient(loss, int_imgs)
                # Add to gradient list
                grads.append(gradient)
            grads = tf.concat(grads, axis = 0)#merge batches into a single set

            # Approximate the integral using the trapezoidal rule
            grads = (grads[:-1] + grads[1:]) / 2.0
            avg_grads = tf.reduce_mean(grads, axis=0)
            # Calculate integrated gradients
            integrated_grads = (image - baseline) * avg_grads
            # Obtain maximum gradient
            integrated_grads = tf.reduce_max(integrated_grads, axis=-1)

            # Convert to NumPy & Remove batch axis
            heatmap = integrated_grads.numpy()
            # Intensity normalization to [0,1]
            min_val = np.amin(heatmap, keepdims = True, axis = tuple(range(1, len(heatmap.shape))))
            max_val = np.amax(heatmap, keepdims = True, axis = tuple(range(1, len(heatmap.shape))))
            numer = heatmap - min_val
            denom = (max_val - min_val) + eps
            hm.append(numer / denom)

        # Return the resulting heatmap
        return np.asarray(hm)
