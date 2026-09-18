# Archived VMamba fine-tuning attempt

This folder is retained for completeness but is not part of the four-model main comparison.

- `pretrained/vmamba_tiny_imagenet.pth` is the original ImageNet-pretrained VMamba-T checkpoint.
- `results_fine_tuned_cpu/` contains the retained fixed-CPU run's final, best-validation-loss, and best-validation-accuracy checkpoints.

The original run did not preserve its notebook, epoch history, predictions, metrics, or split configuration. Therefore these checkpoint-only files should not be reported as a reproducible result. The clean pretrained VMamba experiment is in `../04_vmamba_pretrained/`.
