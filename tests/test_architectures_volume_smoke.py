import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
import torch

from aucmedi.neural_network.architectures.volume import (
    ConvNeXtBase,
    ResNet50,
    ResNet101,
    ResNeXt50,
    ResNeXt101,
    VGG16,
    VGG19,
    MobileNetV2,
    DenseNet121,
    DenseNet201,
    ConvNeXtLarge,
)


def test_volume_architectures_smoke():
    archs = [
        (ConvNeXtBase, (64, 64, 64)),
        (ResNet50, (64, 64, 64)),
        (ResNet101, (64, 64, 64)),
        (ResNeXt50, (64, 64, 64)),
        (ResNeXt101, (64, 64, 64)),
        (DenseNet121, (64, 64, 64)),
        (DenseNet201, (64, 64, 64)),
        (VGG16, (64, 64, 64)),
        (VGG19, (64, 64, 64)),
        (MobileNetV2, (64, 64, 64)),
        (ConvNeXtLarge, (64, 64, 64)),
    ]

    for Arch, res in archs:
        print(f"Testing {Arch.__name__} with input resolution {res}...")
        arch = Arch(channels=3, input_resolution=res, pretrained_weights=False)
        model = arch.create_model()
        model.eval()
        x = torch.randn(2, 3, res[0], res[1], res[2])
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
        assert (
            len(out_t.shape) == 5
        ), f"{Arch.__name__}: output is not 5D -> got={out_t.shape}"
        expected = arch.get_output_shape()
        expected_c = expected[3]
        assert (
            out_t.shape[1] == expected_c
        ), f"{Arch.__name__}: channel mismatch -> got={out_t.shape[1]} expected={expected_c}"

        # assert spatial dimensions are as expected from architecture definition
        expected_h, expected_w, expected_d = expected[:3]
        assert (
            out_t.shape[2] == expected_h
            and out_t.shape[3] == expected_w
            and out_t.shape[4] == expected_d
        ), (
            f"{Arch.__name__}: spatial mismatch -> got={(out_t.shape[2], out_t.shape[3], out_t.shape[4])} "
            f"expected={(expected_h, expected_w, expected_d)}"
        )
