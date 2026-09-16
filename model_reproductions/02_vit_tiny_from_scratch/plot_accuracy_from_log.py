"""Regenerate the two-stage accuracy graph from a completed training log."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
LOG = ROOT / "training_run_corrected.log"
OUTPUT = ROOT / "results" / "two_stage_training_curves.png"

FULL_PATTERN = re.compile(
    r"^Epoch \d+/\d+ .* train loss [\d.]+, acc ([\d.]+) \| "
    r"val loss [\d.]+, acc ([\d.]+)"
)
HEAD_PATTERN = re.compile(
    r"^Head epoch \d+/\d+ \| train loss [\d.]+, acc ([\d.]+) \| "
    r"val loss [\d.]+, acc ([\d.]+)"
)


def read_accuracies():
    full = []
    head = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        match = FULL_PATTERN.match(line)
        if match:
            full.append(tuple(map(float, match.groups())))
            continue
        match = HEAD_PATTERN.match(line)
        if match:
            head.append(tuple(map(float, match.groups())))
    if not full or not head:
        raise RuntimeError(f"Could not find both training stages in {LOG}")
    return full, head


def main():
    full, head = read_accuracies()
    train_color = "#1f77b4"
    validation_color = "#ff7f0e"
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    def plot_phase(axis, points, title):
        epochs = list(range(1, len(points) + 1))
        training = [point[0] for point in points]
        validation = [point[1] for point in points]
        axis.plot(epochs, training, color=train_color, linewidth=2, label="Training")
        axis.plot(
            epochs, validation, color=validation_color, linewidth=2, label="Validation"
        )

        maximum_points = [
            ("Training", training, train_color, -40),
            ("Validation", validation, validation_color, 14),
        ]
        for label, values, color, text_offset in maximum_points:
            index = max(range(len(values)), key=values.__getitem__)
            epoch = epochs[index]
            value = values[index]
            axis.scatter(epoch, value, color=color, s=70, zorder=5, edgecolor="white")
            axis.annotate(
                f"Max {label}: {value:.1%}\nEpoch {epoch}",
                xy=(epoch, value),
                xytext=(0, text_offset),
                textcoords="offset points",
                ha="center",
                color=color,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color},
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.9,
                },
            )

        axis.set(title=title, xlabel="Epoch", ylim=(0, 1.08), xlim=(1, len(epochs)))
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
        return training, validation

    full_training, full_validation = plot_phase(
        axes[0], full, "ViT-Tiny trained from scratch (30 epochs)"
    )
    head_training, head_validation = plot_phase(
        axes[1], head, "ViT-Tiny fine-tuned (30 epochs)"
    )
    axes[0].set_ylabel("Accuracy")
    scratch_best = max(full_validation)
    refined_best = max(head_validation)
    figure.suptitle(
        f"Best observed validation accuracy: {scratch_best:.1%} → {refined_best:.1%} "
        f"({(refined_best - scratch_best) * 100:+.1f} percentage points)"
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {OUTPUT}")
    print(f"Scratch maximum validation accuracy: {scratch_best:.1%}")
    print(f"Classifier-refinement maximum validation accuracy: {refined_best:.1%}")


if __name__ == "__main__":
    main()
