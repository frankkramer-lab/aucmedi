#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
#                    Documentation                    #
#-----------------------------------------------------#
""" The classification variant of the Vision Transformer (ViT) version B32 architecture.

!!! warning
    The ViT architectures only work for RGB encoding (channel size = 3).

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ViT_B32"               |
| Input_shape              | (224, 224)                 |
| Standardization          | "tf"                       |

???+ abstract "Reference - Implementation"
    Fausto Morales; https://github.com/faustomorales <br>
    https://github.com/faustomorales/vit-keras <br>

    Vo Van Tu; https://github.com/tuvovan <br>
    https://github.com/tuvovan/Vision_Transformer_Keras <br>

    Original: Google Research <br>
    https://github.com/google-research/vision_transformer <br>

???+ abstract "Reference - Publication"
    ```
    @article{dosovitskiy2020vit,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
      journal={ICLR},
      year={2021}
    }

    @article{tolstikhin2021mixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision},
      author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
      journal={arXiv preprint arXiv:2105.01601},
      year={2021}
    }

    @article{steiner2021augreg,
      title={How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers},
      author={Steiner, Andreas and Kolesnikov, Alexander and and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
      journal={arXiv preprint arXiv:2106.10270},
      year={2021}
    }

    @article{chen2021outperform,
      title={When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations},
      author={Chen, Xiangning and Hsieh, Cho-Jui and Gong, Boqing},
      journal={arXiv preprint arXiv:2106.01548},
      year={2021},
    }

    @article{zhai2022lit,
      title={LiT: Zero-Shot Transfer with Locked-image Text Tuning},
      author={Zhai, Xiaohua and Wang, Xiao and Mustafa, Basil and Steiner, Andreas and Keysers, Daniel and Kolesnikov, Alexander and Beyer, Lucas},
      journal={CVPR},
      year={2022}
    }
    ```
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from vit_keras import vit
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#     Architecture class: Vision Transformer (ViT)    #
#-----------------------------------------------------#
class ViT_B32(Architecture_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, classification_head, channels, input_shape=(224, 224),
                 pretrained_weights=False):
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights : pretrained = True
        else : pretrained = False

        # Obtain ViT B32 as base model
        base_model = vit.vit_b32(image_size=self.input[:-1],
                                 classes=self.classifier.n_labels,
                                 include_top=False,
                                 pretrained=pretrained,
                                 pretrained_top=False)
        top_model = base_model.output

        # Add classification head
        model = self.classifier.build(model_input=base_model.input,
                                      model_output=top_model)

        # Return created model
        return model
