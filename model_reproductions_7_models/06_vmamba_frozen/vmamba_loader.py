"""Load the genuine VMamba-T architecture saved in this model folder."""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


MODEL_FOLDER = Path(__file__).parent
OFFICIAL_SOURCE = MODEL_FOLDER / "vmamba_official.py"
OFFICIAL_CHECKPOINT = MODEL_FOLDER / "pretrained/vmamba_tiny_imagenet.pth"


class VMambaClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(768, 2)

    def forward_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        return self.head(self.forward_features(images))


def build_vmamba():
    """Construct VMamba from the authors' code and load their ImageNet checkpoint.

    VMamba is not rewritten as a short classroom model because selective scan,
    four-direction scanning and checkpoint-compatible stage construction require
    the full implementation. Replacing them would produce a different model.
    """
    if not OFFICIAL_SOURCE.is_file():
        raise FileNotFoundError(OFFICIAL_SOURCE)
    if not OFFICIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(OFFICIAL_CHECKPOINT)

    spec = importlib.util.spec_from_file_location("local_official_vmamba", OFFICIAL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    backbone = module.vmamba_tiny_s1l8(pretrained=False)
    checkpoint = torch.load(OFFICIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    result = backbone.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"VMamba checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    # Remove the original 1,000-class ImageNet classifier.
    backbone.classifier.head = nn.Identity()
    model = VMambaClassifier(backbone)
    return model, backbone.layers[-1]
