"""DINOv2 ViT-S/14: explicit ViT construction with a frozen backbone."""

from pathlib import Path

from dinov2_model import build_dinov2_small, load_dinov2_pretrained_backbone
from vit_training_helpers import set_seed, train_experiment


MODEL_FOLDER = Path(__file__).parent
DINO_CHECKPOINT = MODEL_FOLDER / "pretrained/dinov2_vits14_pretrain.pth"
set_seed(42)
model = build_dinov2_small()
load_dinov2_pretrained_backbone(model, DINO_CHECKPOINT)

train_experiment(
    model=model,
    phase="frozen",
    last_stage=model.blocks[-1],
    output_dir=MODEL_FOLDER / "results",
    epochs=20,
    learning_rate=1e-4,
    batch_size=16,
    center_crop=True,
)
