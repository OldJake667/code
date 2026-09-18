# Model experiments

The main comparison includes two models trained from scratch and two models using pretrained visual backbones.

1. `01_cnn_from_scratch/cnn_from_scratch.ipynb` — CNN from scratch
2. `02_vit_tiny_from_scratch/vit_tiny_from_scratch.ipynb` — ViT-Tiny/16 from scratch
3. `03_dinov2_pretrained/DINOv2_pretrained.ipynb` — pretrained DINOv2-S/14 with a new two-class classification head
4. `04_vmamba_pretrained/vmamba_pretrained.ipynb` — ImageNet-pretrained VMamba-T with a new two-class classification head

The retained supplementary experiments are in `could_not_run_vmamba_from_scratch`, `no_benefit_dinov2_fine_tuned`, and `no_benefit_vmamba_fine_tuned`.

## What is constructed locally

- The CNN is written layer by layer with ordinary PyTorch layers.
- ViT-Tiny is written layer by layer with ordinary PyTorch layers.
- Pretrained DINOv2-S/14 is loaded through `timm`.
- VMamba uses the model authors' complete local implementation. Rewriting only a short approximation would change selective scan, four-direction scanning, and checkpoint compatibility, so it would no longer be the same VMamba model.
- The pretrained VMamba notebook verifies the checkpoint SHA-256 and downloads the official VMamba-T `s1l8` ImageNet checkpoint from the [authors' GitHub release](https://github.com/MzeroMiko/VMamba/releases/tag/%23v2cls) when it is not already available locally.

## Data split

Every experiment reads the same manifest:

`data/common_split_manifest.csv`

- Training: 176 images
- Validation: 45 images
- Independent test set: none

Therefore, the notebooks report **training accuracy** and **validation accuracy**. They do not call the validation set a test set.

## Checkpoints

The scratch experiments save three complete model checkpoints:

- `final_epoch`: parameters after the scheduled final epoch
- `best_validation_loss`: epoch with the lowest validation loss
- `best_validation_accuracy`: epoch with the highest validation accuracy; ties use the lower loss

The lowest-loss checkpoint is useful because loss contains more information than accuracy when several epochs have the same number of correct images. The best-accuracy checkpoint is also preserved and reported.

The pretrained feature-extractor experiments keep their backbones fixed. Pretrained VMamba therefore saves only the small classification head for each checkpoint; its unchanged ImageNet backbone is stored separately in the same experiment folder.

Selecting a checkpoint and measuring it on the same validation set produces a **best-validation result**, not an independent test result. A future independent test set would be needed for an unbiased final accuracy estimate.
