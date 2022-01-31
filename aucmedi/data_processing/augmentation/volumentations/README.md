# Volumentations 3D

Volumentations is a subpackage of AUCMEDI, which originated from the following Git repositories:
- Original:                 https://github.com/ashawkey/volumentations
- Continued Development:    https://github.com/ZFTurbo/volumentations
- Enhancements:             https://github.com/qubvel/volumentations

Due to a stop of ongoing development in this subpackage, we decided to integrated Volumentations into AUCMEDI to ensure support and functionality.

Nevertheless, if you are using this subpackage, please give credit to all authors including ashawkey, ZFTurbo and qubvel.

Initially inspired by [albumentations](https://github.com/albumentations-team/albumentations) library for augmentation of 2D images.

### Implemented 3D augmentations

```python
PadIfNeeded
GaussianNoise
Resize
RandomScale
RotatePseudo2D
RandomRotate90
Flip
Normalize
Float
Contiguous
Transpose
CenterCrop
RandomResizedCrop
RandomCrop
CropNonEmptyMaskIfExists
ResizedCropNonEmptyMaskIfExists
RandomGamma
ElasticTransformPseudo2D
ElasticTransform
Rotate
RandomCropFromBorders
GridDropout
RandomDropPlane
```

## Credits and License

```
#=================================================================================#
#  Author:       Pavel Iakubovskii, ZFTurbo, ashawkey, Dominik Müller             #
#  Copyright:    Pavel Iakubovskii  : https://github.com/qubvel                   #
#                ZFTurbo            : https://github.com/ZFTurbo                  #
#                ashawkey           : https://github.com/ashawkey                 #
#                Dominik Müller     : https://github.com/muellerdo                #
#                2022 IT-Infrastructure for Translational Medical Research,       #
#                University of Augsburg                                           #
#                                                                                 #
#  Volumentations is a subpackage of AUCMEDI, which originated from the           #
#  following Git repositories:                                                    #
#       - Original:                 https://github.com/ashawkey/volumentations    #
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
```

**Also please cite the paper from ZFTurbo:**  
More details on ArXiv: https://arxiv.org/abs/2104.01687
```
@InProceedings{RSolovyev_2021_stalled,
  author = {Solovyev, Roman and Kalinin, Alexandr A. and Gabruseva, Tatiana},
  title = {3D Convolutional Neural Networks for Stalled Brain Capillary Detection},
  booktitle = {Arxiv: 2104.01687},
  month = {April},
  year = {2021}
}
```
