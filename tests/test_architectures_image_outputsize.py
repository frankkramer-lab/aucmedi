import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
import unittest
import torch

from aucmedi import *
from aucmedi.neural_network.architectures.image import (
    ResNet50,
    ResNet101,
    ResNeXt50,
    ResNeXt101,
    VGG16,
    VGG19,
    MobileNetV2,
    EfficientNetB0,
    EfficientNetB1,
    ViT_B16,
    ViT_L16,
    ViT_B32,
    ViT_L32,
    DenseNet121,
    DenseNet201,
    InceptionV3,
    ConvNeXtLarge,
    ConvNeXtBase,
    ConvNeXtTiny,
)


class ArchitecturesImageOutputSizeTEST(unittest.TestCase):
    def test_image_architectures(self):
        archs = [
            (ResNet50, (250, 250)),
            (ResNet101, (250, 250)),
            (ResNeXt50, (250, 250)),
            (ResNeXt101, (250, 250)),
            (DenseNet121, (250, 250)),
            (DenseNet201, (250, 250)),
            (InceptionV3, (250, 250)),
            (VGG16, (250, 250)),
            (VGG19, (250, 250)),
            (MobileNetV2, (250, 250)),
            (InceptionV3, (299, 299)),
            (EfficientNetB0, (250, 250)),
            (EfficientNetB1, (250, 250)),
            (ViT_B16, (250, 250)),
            (ViT_L16, (250, 250)),
            (ViT_B32, (250, 250)),
            (ViT_L32, (250, 250)),
            (ConvNeXtTiny, (250, 250)),
            (ConvNeXtBase, (250, 250)),
            (ConvNeXtLarge, (553, 553)),
        ]

        for Arch, res in archs:
            num_channels = 3
            arch = Arch(
                channels=num_channels, input_resolution=res, pretrained_weights=False
            )
            model = arch.create_model()
            model.eval()
            x = torch.randn(1, num_channels, res[0], res[1])
            with torch.no_grad():
                out = model(x)

            # feature extractor might return tensor or dict
            out_t = None
            if isinstance(out, dict):
                out_t = list(out.values())[0]
            else:
                out_t = out

            # assert channels as expected from architecture definition
            assert hasattr(out_t, "shape"), f"{Arch.__name__}: output has no shape"
            expected = arch.get_output_shape()
            expected_c = expected[2]
            assert (
                out_t.shape[1] == expected_c
            ), f"{Arch.__name__}: channel mismatch -> got={out_t.shape[1]} expected={expected_c}"

            # assert spatial dimensions are as expected from architecture definition
            expected_h, expected_w = expected[:2]
            assert out_t.shape[2] == expected_h and out_t.shape[3] == expected_w, (
                f"{Arch.__name__}: spatial mismatch -> got={(out_t.shape[2], out_t.shape[3])} "
                f"expected={(expected_h, expected_w)}"
            )
