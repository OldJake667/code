"""VMamba-T: keep the genuine pretrained VMamba backbone frozen."""

from pathlib import Path

from vit_training_helpers import set_seed, train_frozen_with_feature_cache
from vmamba_loader import build_vmamba


MODEL_FOLDER = Path(__file__).parent
set_seed(42)
model, last_stage = build_vmamba()
train_frozen_with_feature_cache(
    model=model,
    output_dir=MODEL_FOLDER / "results",
    epochs=20,
    learning_rate=1e-4,
    image_batch_size=2,
    classifier_batch_size=16,
    center_crop=True,
)
