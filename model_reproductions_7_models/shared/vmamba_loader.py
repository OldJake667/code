"""Construct the genuine VMamba-T architecture with random weights."""

import importlib.util
import sys
from pathlib import Path

import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SOURCE = PROJECT_ROOT / "model_reproductions_7_models/06_vmamba_frozen/vmamba_official.py"


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
    """Construct VMamba-T from the authors' code without loading a checkpoint.

    VMamba is not rewritten as a short classroom model because selective scan,
    four-direction scanning and checkpoint-compatible stage construction require
    the full implementation. Replacing them would produce a different model.
    """
    if not OFFICIAL_SOURCE.is_file():
        raise FileNotFoundError(OFFICIAL_SOURCE)
    spec = importlib.util.spec_from_file_location("local_official_vmamba", OFFICIAL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    backbone = module.vmamba_tiny_s1l8(pretrained=False)
    # Remove the randomly initialized 1,000-class head and add a random 2-class head.
    backbone.classifier.head = nn.Identity()
    model = VMambaClassifier(backbone)
    return model, backbone.layers[-1]
