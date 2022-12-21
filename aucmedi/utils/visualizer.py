#==============================================================================#
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
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import SimpleITK as sitk
# Internal libraries
from aucmedi.data_processing.subfunctions import Standardize

#-----------------------------------------------------#
#                   Image Visualizer                  #
#-----------------------------------------------------#
def visualize_image(array, out_path=None):
    """ Simple wrapper function to visualize a NumPy matrix as image via PIL.

    ???+ info
        NumPy array shape has to be (x, y, channel) like this: (224, 224, 3)

    Args:
        array (numpy.ndarray):          NumPy matrix containing an image.
        out_path (str):                 Path in which image is stored (else live output).
    """
    # Ensure integer intensity values
    array = np.uint8(array)
    # Remove channel axis if grayscale
    if array.shape[-1] == 1 : array = np.reshape(array, array.shape[:-1])
    # Convert array to PIL image
    image = Image.fromarray(array)
    # Visualize or store image
    if out_path is None : image.show()
    else : image.save(out_path)

#-----------------------------------------------------#
#                  Volume Visualizer                  #
#-----------------------------------------------------#
def visualize_volume(array, out_path=None, iteration_axis=1):
    """ Simple wrapper function to visualize/store a NumPy matrix as volume.

    ???+ info
        NumPy array shape has to be (x, y, z, channel) like this: (128, 128, 128, 1)

    Args:
        array (numpy.ndarray):          NumPy matrix containing an image.
        out_path (str):                 Path in which volume is stored (else live output).
    """
    # Identify output format
    if out_path is None : format = "gif"
    elif out_path.split(".")[-1].lower() in ["gif", "mha", "nii", "gz", "npy"]:
        format = out_path.split(".")[-1]
    else : raise ValueError("Visualizer does not support image format!")

    # Encode as GIF
    if format == "gif":
        # Grayscale normalization if required
        if array.shape[-1] == 1 and (np.min(array) < 0 or np.max(array) > 255):
            array = Standardize(mode="grayscale").transform(array)
        elif array.shape[-1] != 1 and (np.min(array) < 0 or np.max(array) > 255):
            raise ValueError("Visualizer does not support multi-channel " + \
                            "non-RGB formats." + \
                            " Array min: " + str(np.min(array)) + \
                            " Array max: " + str(np.max(array))) 
        # Ensure integer intensity values
        array = np.uint8(array)

        # Create a figure and two axes objects from matplot
        fig = plt.figure()
        img = plt.imshow(np.take(array, 0, axis=iteration_axis),
                        cmap='gray', vmin=0, vmax=255, animated=True)
        # Update function to show the slice for the current frame
        def update(i):
            plt.suptitle("Slice: " + str(i))
            img.set_data(np.take(array, i, axis=iteration_axis))
            return img
        # Compute the animation (gif)
        ani = animation.FuncAnimation(fig, update, 
                                    frames=array.shape[iteration_axis],
                                    interval=5, 
                                    repeat_delay=0, 
                                    blit=False)
        # Visualize or store gif
        if out_path is None : plt.show()
        else : ani.save(out_path, writer='imagemagick', fps=None, dpi=None)
        # Close the matplot
        plt.close()
    # Encode as NumPy file
    elif format == "npy" : np.save(out_path, array)
    # Encode as ITK file
    else:
        array_sitk = sitk.GetImageFromArray(array)
        sitk.WriteImage(array_sitk, out_path)

#-----------------------------------------------------#
#               XAI Heatmap Visualizer                #
#-----------------------------------------------------#
def visualize_heatmap(image, heatmap, overlay=True, out_path=None, alpha=0.4):
    """ Simple wrapper function to visualize a heatmap encoded as NumPy matrix with a
        [0-1] range as image/volume via matplotlib.

    ??? abstract "Reference - Implementation"
        Author: François Chollet <br>
        Date: April 26, 2020 <br>
        https://keras.io/examples/vision/grad_cam/ <br>

    Args:
        image (numpy.ndarray):          NumPy matrix containing an image or volume.
        heatmap (numpy.ndarray):        NumPy matrix containing a XAI heatmap.
        out_path (str):                 Path in which image is stored (else live output).
        alpha (float):                  Transparency value for heatmap overlap on image (range: [0-1]).
    """
    # Grayscale normalization if required
    if image.shape[-1] == 1 and (np.min(image) < 0 or np.max(image) > 255):
        image = Standardize(mode="grayscale").transform(image)
    elif image.shape[-1] != 1 and (np.min(image) < 0 or np.max(image) > 255):
        raise ValueError("Visualizer does not support multi-channel " + \
                         "non-RGB formats." + \
                         " Array min: " + str(np.min(image)) + \
                         " Array max: " + str(np.max(image))) 
    # If image is grayscale, convert to RGB
    if image.shape[-1] == 1 : image = np.concatenate((image,)*3, axis=-1)
    # Rescale heatmap to grayscale range
    heatmap = np.uint8(heatmap * 255)
    # Overlay the heatmap on the image 
    if overlay:
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:,:3]
        jet_heatmap = jet_colors[heatmap] * 255
        # Superimpose the heatmap on original image
        final_img = jet_heatmap * alpha + (1-alpha) * image
    # Output just the heatmap
    else : final_img = heatmap
    # Apply corresponding 2D visualizer 
    if len(image.shape) == 3 : visualize_image(final_img, out_path=out_path)
    elif len(image.shape) == 4 : visualize_volume(final_img, out_path=out_path)