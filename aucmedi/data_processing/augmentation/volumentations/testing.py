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

# -----------------------------------------------------#
#                    GIF Visualizer                    #
# -----------------------------------------------------#
def visualize_evaluation(index, volume, viz_path="test_volumentations"):
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
    from skimage.data import brain
    ds = brain()

    print(ds.shape)
