#=================================================================================#
#  Author:       Pavel Iakubovskii, ZFTurbo, ashawkey, Dominik Müller             #
#  Copyright:    albumentations:    : https://github.com/albumentations-team      #
#                Pavel Iakubovskii  : https://github.com/qubvel                   #
#                ZFTurbo            : https://github.com/ZFTurbo                  #
#                ashawkey           : https://github.com/ashawkey                 #
#                Dominik Müller     : https://github.com/muellerdo                #
#                2022 IT-Infrastructure for Translational Medical Research,       #
#                University of Augsburg                                           #
#                                                                                 #
#  Volumentations is a subpackage of AUCMEDI, which originated from the           #
#  following Git repositories:                                                    #
#       - Original:                 https://github.com/albumentations-team/album  #
#                                   entations                                     #
#       - 3D Conversion:            https://github.com/ashawkey/volumentations    #
#       - Continued Development:    https://github.com/ZFTurbo/volumentations     #
#       - Enhancements:             https://github.com/qubvel/volumentations      #
#                                                                                 #
#  Due to a stop of ongoing development in this subpackage, we decided to         #
#  integrated Volumentations into AUCMEDI to ensure support and functionality.    #
#                                                                                 #
#  MIT License.                                                                   #
#                                                                                 #
#  Permission is hereby granted, free of charge, to any person obtaining a copy   #
#  of this software and associated documentation files (the "Software"), to deal  #
#  in the Software without restriction, including without limitation the rights   #
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
#  copies of the Software, and to permit persons to whom the Software is          #
#  furnished to do so, subject to the following conditions:                       #
#                                                                                 #
#  The above copyright notice and this permission notice shall be included in all #
#  copies or substantial portions of the Software.                                #
#                                                                                 #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
#  SOFTWARE.                                                                      #
#=================================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.data import cells3d
# AUCMED libraries
from aucmedi.data_processing.augmentation import Volume_Augmentation

# -----------------------------------------------------#
#                    GIF Visualizer                    #
# -----------------------------------------------------#
def grayscale_normalization(image):
    # Identify minimum and maximum
    max_value = np.max(image)
    min_value = np.min(image)
    # Scaling
    image_scaled = (image - min_value) / (max_value - min_value)
    image_normalized = np.around(image_scaled * 255, decimals=0)
    # Return normalized image
    return image_normalized

def visualize_evaluation(index, volume, viz_path="test_volumentations"):
    # Grayscale Normalization of Volume
    volume_gray = grayscale_normalization(volume)

    # Create a figure and two axes objects from matplot
    fig = plt.figure()
    img = plt.imshow(volume_gray[0, :, :], cmap='gray', vmin=0, vmax=255,
                     animated=True)

    # Update function to show the slice for the current frame
    def update(i):
        plt.suptitle("Augmentation: " + str(index) + " - " + "Slice: " + str(i))
        img.set_data(volume_gray[i, :, :])
        return img

    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=volume_gray.shape[0],
                                  interval=5, repeat_delay=0, blit=False)
    # Set up the output path for the gif
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)
    file_name = "visualization." + str(index) + ".gif"
    out_path = os.path.join(viz_path, file_name)
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=None, dpi=None)
    # Close the matplot
    plt.close()

#-----------------------------------------------------#
#                  Application Test                   #
#-----------------------------------------------------#
if __name__ == "__main__":
    # Obtain 3D volume of fluorescence microscopy image of cells
    data_raw = cells3d()
    # Extract nuclei
    data = np.reshape(data_raw[:,1,:,:], (60, 256, 256))
    data = np.float32(data)
    data = grayscale_normalization(data)
    # Visualize original volume
    visualize_evaluation("original", data)
    print(data)
    print("original", data.shape)
    # Setup options
    options = [False for x in range(15)]
    options_labels = ["flip", "rotate", "brightness", "contrast", "saturation",
                      "hue", "scale", "crop", "grid_distortion", "compression",
                      "gaussian_noise", "gaussian_blur", "downscaling", "gamma",
                      "elastic_transform"]
    # Apply each augmentation once for testing
    for i in range(15):
        # Active current augmentation technique
        options_curr = options.copy()
        options_curr[i] = True
        # Initialize Volumentations
        data_aug = Volume_Augmentation(*options_curr)
        data_aug.aug_elasticTransform_p = 1.0
        data_aug.aug_gamma_p = 1.0
        data_aug.aug_downscaling_p = 1.0
        data_aug.aug_gaussianBlur_p = 1.0
        data_aug.aug_gaussianNoise_p = 1.0
        data_aug.aug_compression_p = 1.0
        data_aug.aug_gridDistortion_p = 1.0
        data_aug.aug_crop_p = 1.0
        data_aug.aug_scale_p = 1.0
        data_aug.aug_hue_p = 1.0
        data_aug.aug_saturation_p = 1.0
        data_aug.aug_contrast_p = 1.0
        data_aug.aug_brightness_p = 1.0
        data_aug.aug_rotate_p = 1.0
        data_aug.aug_flip_p = 1.0
        data_aug.aug_crop_shape = (30, 64, 64)
        data_aug.aug_compression_limits = (100,100)
        data_aug.build()
        # Apply augmentation
        img_augmented = data_aug.apply(data)
        # Visualize result
        print(img_augmented)
        print(options_labels[i], img_augmented.shape)
        visualize_evaluation(options_labels[i], img_augmented)
