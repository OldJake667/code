"""DINOv2: start from the best frozen checkpoint and tune block 12 + head."""

from pathlib import Path

from dinov2_model import build_dinov2_small
from vit_training_helpers import set_seed, train_experiment


MODEL_FOLDER = Path(__file__).parent
set_seed(42)
model = build_dinov2_small()
initial_checkpoint = MODEL_FOLDER.parent / "04_dinov2_frozen/results/best_validation_loss.pth"

train_experiment(
    model=model,
    phase="fine_tuned",
    last_stage=model.blocks[-1],
    initial_checkpoint=initial_checkpoint,
    output_dir=MODEL_FOLDER / "results",
    epochs=10,
    learning_rate=1e-5,
    batch_size=8,
    center_crop=True,
)
