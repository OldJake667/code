"""VMamba-T: start from the best frozen checkpoint and tune its final stage."""

from pathlib import Path

from vit_training_helpers import set_seed, train_experiment
from vmamba_loader import build_vmamba


MODEL_FOLDER = Path(__file__).parent
set_seed(42)
model, last_stage = build_vmamba()
initial_checkpoint = MODEL_FOLDER.parent / "06_vmamba_frozen/results/best_validation_loss.pth"

train_experiment(
    model=model,
    phase="fine_tuned",
    last_stage=last_stage,
    initial_checkpoint=initial_checkpoint,
    output_dir=MODEL_FOLDER / "results",
    epochs=10,
    learning_rate=1e-5,
    batch_size=1,
    center_crop=True,
)
