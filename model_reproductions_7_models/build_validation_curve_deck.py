"""Build a presentation of validation accuracy at every training epoch."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "presentation"
CURVE_DIR = OUT / "validation_curves"
CURVE_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        "name": "CNN from scratch",
        "short": "cnn",
        "source": ROOT / "01_cnn_from_scratch/training_history.json",
        "phase": "20-epoch training run",
        "color": "28689D",
        # Transcribed from the CNN chart embedded in
        # /Users/olzhi/Downloads/Presentation 28.08.26.pptx. With 45 validation
        # images, each plotted level maps exactly to an integer correct count.
        "validation_correct": [
            37, 40, 41, 40, 42, 40, 41, 40, 42, 40,
            40, 41, 40, 41, 40, 42, 42, 41, 42, 41,
        ],
    },
    {
        "name": "ViT-Tiny frozen",
        "short": "vit_tiny_frozen",
        "source": ROOT / "02_vit_tiny_frozen/notebook_results_run2/training_history.csv",
        "phase": "20-epoch frozen-backbone run",
        "color": "28689D",
    },
    {
        "name": "ViT-Tiny fine-tuned",
        "short": "vit_tiny_fine_tuned",
        "source": ROOT / "03_vit_tiny_fine_tuned/notebook_results_cpu_notebook/training_history.csv",
        "phase": "10-epoch fine-tuning run",
        "color": "188C87",
    },
    {
        "name": "DINOv2 frozen",
        "short": "dinov2_frozen",
        "source": ROOT / "04_dinov2_frozen/notebook_results/training_history.csv",
        "phase": "20-epoch frozen-backbone run",
        "color": "28689D",
    },
    {
        "name": "DINOv2 fine-tuned",
        "short": "dinov2_fine_tuned",
        "source": ROOT / "05_dinov2_fine_tuned/notebook_results_cpu_notebook/training_history.csv",
        "phase": "10-epoch fine-tuning run",
        "color": "188C87",
    },
    {
        "name": "VMamba frozen",
        "short": "vmamba_frozen",
        "source": ROOT / "06_vmamba_frozen/notebook_results/training_history.csv",
        "phase": "20-epoch frozen-backbone run",
        "color": "28689D",
    },
    {
        "name": "VMamba fine-tuned",
        "short": "vmamba_fine_tuned",
        "source": ROOT / "07_vmamba_fine_tuned/notebook_results_fixed_cpu/training_history.csv",
        "phase": "10-epoch fine-tuning run",
        "color": "188C87",
    },
]

BG = "F5F7FA"
NAVY = "152536"
TEXT = "344657"
MUTED = "667788"
WHITE = "FFFFFF"
ORANGE = "E1843B"


def load_history(experiment):
    if "validation_correct" in experiment:
        values = [100 * correct / 45 for correct in experiment["validation_correct"]]
        return list(range(1, len(values) + 1)), values

    path = experiment["source"]
    if path.suffix == ".json":
        history = json.loads(path.read_text(encoding="utf-8"))
        values = [100 * float(value) for value in history["val_accuracy"]]
        epochs = list(range(1, len(values) + 1))
    else:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in rows]
        values = [100 * float(row["validation_accuracy"]) for row in rows]
    return epochs, values


def make_curve(experiment):
    epochs, values = load_history(experiment)
    color = f"#{experiment['color']}"
    best_value = max(values)
    best_index = values.index(best_value)
    final_value = values[-1]

    figure, axis = plt.subplots(figsize=(12.2, 5.25))
    axis.plot(
        epochs,
        values,
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=2,
    )
    axis.scatter(
        [epochs[best_index]],
        [best_value],
        s=115,
        color=f"#{ORANGE}",
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
        label=f"Best: {best_value:.2f}% at epoch {epochs[best_index]}",
    )

    lower = max(0, 5 * int((min(values) - 6) / 5))
    axis.set_ylim(lower, 102)
    axis.set_xlim(.5, len(epochs) + .5)
    axis.set_xlabel("Epoch", fontsize=13)
    axis.set_ylabel("Validation accuracy (%)", fontsize=13)
    axis.set_xticks(epochs)
    axis.tick_params(axis="x", labelsize=10)
    axis.tick_params(axis="y", labelsize=10, length=0)
    axis.grid(axis="both", alpha=.18)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right", fontsize=11)

    # Label the final point separately when it is not the best point.
    if best_index != len(values) - 1:
        axis.annotate(
            f"Final: {final_value:.2f}%",
            (epochs[-1], final_value),
            xytext=(-8, -22 if final_value > 95 else 12),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            fontweight="bold",
            color=f"#{NAVY}",
        )

    figure.tight_layout()
    path = CURVE_DIR / f"{experiment['short']}_validation_accuracy.png"
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path, epochs, values


def rgb(value):
    return RGBColor.from_string(value)


def add_text(slide, value, x, y, w, h, size=18, color=TEXT, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(.03)
    frame.margin_top = frame.margin_bottom = Inches(.02)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_curve_slide(prs, experiment, chart, epochs, values):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(BG)
    add_text(slide, experiment["phase"].upper(), .68, .2, 12, .25, 10, experiment["color"], True)
    add_text(slide, experiment["name"], .68, .5, 12, .55, 27, NAVY, True)
    slide.shapes.add_picture(str(chart), Inches(.62), Inches(1.15), width=Inches(12.1))
    add_text(
        slide,
        f"Every saved epoch is shown • {len(epochs)} epochs • final validation accuracy {values[-1]:.2f}%",
        .7, 7.08, 11.95, .25, 10, MUTED, False, PP_ALIGN.CENTER,
    )


def add_reported_score_slide(prs, experiment):
    correct = experiment["reported_correct"]
    percent = 100 * correct / 45
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(BG)
    add_text(slide, experiment["phase"].upper(), .68, .2, 12, .25, 10, experiment["color"], True)
    add_text(slide, experiment["name"], .68, .5, 12, .55, 27, NAVY, True)
    add_text(slide, f"{correct}/45", .78, 1.6, 11.8, 1.15, 48, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, f"{percent:.2f}% validation accuracy", .78, 2.75, 11.8, .6, 23, experiment["color"], True, PP_ALIGN.CENTER)
    add_text(
        slide,
        "This is the CNN result retained in the earlier comparison presentation. The matching per-epoch history file is no longer present, so a 20-epoch curve cannot be reconstructed reliably.",
        1.45, 4.15, 10.4, 1.15, 17, TEXT, False, PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "The separate CNN notebook history that reaches 45/45 belongs to a different run and is intentionally not mixed with this result.",
        1.45, 5.55, 10.4, .72, 13, MUTED, False, PP_ALIGN.CENTER,
    )


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[6])
fill = slide.background.fill
fill.solid()
fill.fore_color.rgb = rgb(NAVY)
add_text(slide, "PAPER REPLICATION", .78, .62, 11.8, .35, 11, "6DD1CA", True)
add_text(slide, "Validation accuracy by epoch", .78, 1.45, 11.8, .9, 34, WHITE, True)
add_text(slide, "Seven saved validation-accuracy curves", .8, 2.55, 11.7, .55, 22, "B9D4E5")
add_text(slide, "Y-axis: validation accuracy  •  X-axis: epoch", .8, 5.75, 11.7, .5, 17, WHITE)
add_text(slide, "31 August 2026", .8, 6.38, 11.7, .35, 12, "A6B8C8")

for experiment in EXPERIMENTS:
    chart, epochs, values = make_curve(experiment)
    add_curve_slide(prs, experiment, chart, epochs, values)

output = OUT / "validation_accuracy_by_epoch_20260831.pptx"
prs.save(output)
print(output)
