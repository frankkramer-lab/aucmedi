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
#                    Documentation                    #
# -----------------------------------------------------#
"""The classification variant of the Vision Transformer (ViT) version L32 architecture.

!!! warning
    The ViT architectures only work for RGB encoding (channel size = 3).

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.ViT_L32"               |
| Input_shape              | (384, 384)                 |
| Standardization          | "torch"                    |

Choose pretrained weights via the torchvision `ViT_L_32_Weights` enum and use
the `get_preprocess()` helper to obtain the correct preprocessing transforms.

???+ abstract "Reference - Implementation"
  [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_l_32.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_l_32.html) <br>

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

# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
from torchvision.models import vit_l_32 as TorchvisionModel
from torchvision.models import ViT_L_32_Weights
import torch
import torch.nn as nn

# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base


# -----------------------------------------------------#
#     Architecture class: Vision Transformer (ViT)    #
# -----------------------------------------------------#
class ViT_L32(Architecture_Base):
    def __init__(self, channels, input_resolution=(384, 384), pretrained_weights=False):
        self.input_shape = input_resolution + (channels,)
        self.pretrained_weights = pretrained_weights
        self.channels = channels

    def get_output_shape(self):
        return (1, 1, 1024)

    def get_preprocess(self):
        weights = ViT_L_32_Weights.DEFAULT
        return weights.transforms()

    def create_model(self):
        if self.pretrained_weights:
            weights_arg = ViT_L_32_Weights.DEFAULT
        else:
            weights_arg = None
        # Instantiate model with a patch-aligned image size and remove head
        patch_size = 32
        h_in = self.input_shape[0]
        w_in = self.input_shape[1]
        target_h = ((h_in + patch_size - 1) // patch_size) * patch_size
        target_w = ((w_in + patch_size - 1) // patch_size) * patch_size
        target_size = max(target_h, target_w)

        full_model = TorchvisionModel(weights=weights_arg, image_size=target_size)
        try:
            full_model.heads = nn.Identity()
        except Exception:
            if hasattr(full_model, "heads") and hasattr(full_model.heads, "head"):
                full_model.heads.head = nn.Identity()

        class _PadAndRun(nn.Module):
            def __init__(self, model, target_h, target_w):
                super().__init__()
                self.model = model
                self.target_h = target_h
                self.target_w = target_w

            def forward(self, x):
                _, _, h, w = x.shape
                pad_h = max(0, self.target_h - h)
                pad_w = max(0, self.target_w - w)
                if pad_h != 0 or pad_w != 0:
                    import torch.nn.functional as F

                    x = F.pad(x, (0, pad_w, 0, pad_h))
                out = self.model(x)
                if isinstance(out, dict):
                    out = list(out.values())[0]
                if hasattr(out, "ndim") and out.ndim == 2:
                    out = out.unsqueeze(-1).unsqueeze(-1)
                return out

        return _PadAndRun(full_model, target_size, target_size)
