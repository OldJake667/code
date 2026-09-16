# Model experiments

The comparison includes models trained from scratch and DINOv2 experiments initialized from pretrained weights.

1. `01_cnn_from_scratch/cnn code.ipynb` — CNN from scratch
2. `02_vit_tiny_from_scratch/vit_tiny_from_scratch.ipynb` — ViT-Tiny/16 from scratch
3. `03_dinov2_pretrained/DINOv2_pretrained.ipynb` — pretrained DINOv2-S/14 and an independent classifier-only Fine-tuned DINOv2 run
4. `04_vmamba_from_scratch/VMamba_from_scratch.ipynb` — VMamba-T from scratch

## What is constructed locally

- The CNN is written layer by layer in TensorFlow.
- ViT-Tiny and the ViT-S/14 architecture are written layer by layer with ordinary PyTorch layers.
- VMamba uses the model authors' complete local implementation. Rewriting only a short approximation would change selective scan, four-direction scanning and checkpoint compatibility, so it would no longer be the same VMamba model.

## Data split

Every experiment reads the same manifest:

`data/common_split_manifest.csv`

- Training: 176 images
- Validation: 45 images
- Independent test set: none

Therefore, the notebooks report **training accuracy** and **validation accuracy**. They do not call the validation set a test set.

## Checkpoints

Every new run saves three checkpoints:

- `final_epoch`: parameters after the scheduled final epoch
- `best_validation_loss`: epoch with the lowest validation loss
- `best_validation_accuracy`: epoch with the highest validation accuracy; ties use the lower loss

The fine-tuning notebooks start from `best_validation_loss`. This is a normal model-selection checkpoint because loss contains more information than accuracy when several epochs have the same number of correct images. The best-accuracy checkpoint is also preserved and reported.

Selecting a checkpoint and measuring it on the same validation set produces a **best-validation result**, not an independent test result. A future independent test set would be needed for an unbiased final accuracy estimate.

## Kernels

- CNN notebook: `/Users/olzhi/Desktop/Paper replication/.venv/bin/python`
- ViT, DINOv2 and VMamba notebooks: `/Users/olzhi/Documents/Research/Research 2026-07-15/.venv/bin/python`

Run each notebook independently; the DINOv2 notebook downloads its pretrained `timm` weights on the first run.
