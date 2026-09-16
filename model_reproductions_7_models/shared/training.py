"""Shared data loading, training, checkpointing and evaluation code."""

import csv
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


CLASS_NAMES = ["Low", "High"]
MANIFEST = Path(__file__).resolve().parents[2] / "data/common_split_manifest.csv"


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_common_split():
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    train = []
    validation = []
    for row in rows:
        path = Path(row["full_path"])
        if not path.is_file():
            folder = "combined_high_bw" if int(row["label"]) else "combined_low_bw"
            # The manifest may retain absolute paths from the computer on which
            # it was created. Resolve moved datasets relative to the manifest.
            path = MANIFEST.parent / folder / row["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        item = (path, int(row["label"]), row["file"])
        (validation if row["split"] == "validation" else train).append(item)

    assert len(rows) == 221
    assert len(train) == 176
    assert len(validation) == 45
    assert not ({x[2] for x in train} & {x[2] for x in validation})
    return train, validation


class GeometryDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, filename = self.samples[index]
        image = Image.open(path).convert("L").convert("RGB")
        return self.transform(image), label, filename


def make_loaders(batch_size, center_crop):
    train, validation = read_common_split()
    resize = [transforms.Resize(256), transforms.CenterCrop(224)] if center_crop else [transforms.Resize((224, 224))]
    transform = transforms.Compose([
        *resize,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        GeometryDataset(train, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        GeometryDataset(validation, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader


def configure_parameters(model, phase, last_stage):
    if phase in {"from_scratch", "pretrained_all_parameters"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False

    if phase == "fine_tuned":
        for parameter in last_stage.parameters():
            parameter.requires_grad = True

    for parameter in model.head.parameters():
        parameter.requires_grad = True


def one_epoch(model, loader, loss_function, device, optimizer=None, last_stage=None):
    training = optimizer is not None
    model.eval()
    if training:
        if last_stage is model:
            model.train()
        else:
            model.head.train()
        if last_stage is not None and last_stage is not model:
            last_stage.train()

    total_loss = 0.0
    correct = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            scores = model(images)
            loss = loss_function(scores, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += int((scores.argmax(1) == labels).sum())

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    labels_out, predictions, probabilities, filenames = [], [], [], []
    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device)
            labels = labels.to(device)
            scores = model(images)
            probs = torch.softmax(scores, dim=1)
            total_loss += loss_function(scores, labels).item() * len(labels)
            labels_out.extend(labels.cpu().tolist())
            predictions.extend(scores.argmax(1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            filenames.extend(names)

    matrix = confusion_matrix(labels_out, predictions, labels=[0, 1])
    report = classification_report(
        labels_out,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    correct = sum(a == b for a, b in zip(labels_out, predictions))
    metrics = {
        "validation_loss": total_loss / len(loader.dataset),
        "validation_accuracy": correct / len(loader.dataset),
        "correct_predictions": correct,
        "incorrect_predictions": len(loader.dataset) - correct,
        "prediction_count": len(loader.dataset),
        "confusion_matrix": matrix.tolist(),
        "Low": report["Low"],
        "High": report["High"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    return metrics, list(zip(filenames, labels_out, predictions, probabilities))


def save_history(history, output_dir):
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(epochs, [row["train_accuracy"] for row in history], "o-", label="Training")
    axes[0].plot(epochs, [row["validation_accuracy"] for row in history], "o-", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].plot(epochs, [row["train_loss"] for row in history], "o-", label="Training")
    axes[1].plot(epochs, [row["validation_loss"] for row in history], "o-", label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=220)
    plt.close(figure)


def save_evaluation(name, model, loader, loss_function, device, output_dir):
    metrics, predictions = evaluate(model, loader, loss_function, device)
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{name}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "true_label", "predicted_label", "probability_low", "probability_high", "correct"])
        for filename, true, predicted, probs in predictions:
            writer.writerow([filename, true, predicted, probs[0], probs[1], true == predicted])
    return metrics


def train_experiment(
    model,
    phase,
    last_stage,
    output_dir,
    epochs,
    learning_rate,
    batch_size,
    center_crop,
    initial_checkpoint=None,
):
    """Train and save final, best-loss and best-accuracy checkpoints."""
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = choose_device()

    if initial_checkpoint:
        state = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    configure_parameters(model, phase, last_stage)
    model.to(device)
    train_loader, validation_loader = make_loaders(batch_size, center_crop)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []
    best_loss = float("inf")
    best_accuracy = -1.0
    best_accuracy_loss = float("inf")
    for epoch in range(1, epochs + 1):
        training_stage = (
            model
            if phase in {"from_scratch", "pretrained_all_parameters"}
            else (last_stage if phase == "fine_tuned" else None)
        )
        train_loss, train_accuracy = one_epoch(
            model, train_loader, loss_function, device, optimizer, training_stage
        )
        validation_loss, validation_accuracy = one_epoch(
            model, validation_loader, loss_function, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_validation_loss.pth")

        if validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_accuracy_loss
        ):
            best_accuracy = validation_accuracy
            best_accuracy_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_validation_accuracy.pth")

    torch.save(model.state_dict(), output_dir / "final_epoch.pth")
    save_history(history, output_dir)

    summary = {}
    for name, filename in (
        ("final", "final_epoch.pth"),
        ("best_loss", "best_validation_loss.pth"),
        ("best_accuracy", "best_validation_accuracy.pth"),
    ):
        model.load_state_dict(torch.load(output_dir / filename, map_location=device, weights_only=True))
        summary[name] = save_evaluation(
            name, model, validation_loader, loss_function, device, output_dir
        )

    configuration = {
        "phase": phase,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "seed": 42,
        "training_images": 176,
        "validation_images": 45,
        "test_images": 0,
        "test_set_note": "No independent test set exists in the authoritative 176/45 split.",
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    (output_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def _feature_epoch(head, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    correct = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            scores = head(features)
            loss = loss_function(scores, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += int((scores.argmax(1) == labels).sum())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def _extract_fixed_features(model, loader, device):
    """Run a frozen backbone once; its output cannot change between epochs."""
    model.eval()
    all_features, all_labels = [], []
    with torch.inference_mode():
        for images, labels, _ in loader:
            features = model.forward_features(images.to(device))
            # timm Vision Transformers return every token from forward_features.
            # Convert those tokens to the same pre-logit vector consumed by head.
            if features.ndim == 3 and hasattr(model, "forward_head"):
                features = model.forward_head(features, pre_logits=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def train_frozen_with_feature_cache(
    model,
    output_dir,
    epochs=20,
    learning_rate=1e-4,
    image_batch_size=2,
    classifier_batch_size=16,
    center_crop=True,
):
    """Efficient frozen training: extract fixed features once, then train the head."""
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = choose_device()
    configure_parameters(model, "frozen", last_stage=None)
    model.to(device)

    image_train, image_validation = make_loaders(image_batch_size, center_crop)
    train_features, train_labels = _extract_fixed_features(model, image_train, device)
    validation_features, validation_labels = _extract_fixed_features(model, image_validation, device)
    torch.save({"features": train_features, "labels": train_labels}, output_dir / "cached_train_features.pt")
    torch.save(
        {"features": validation_features, "labels": validation_labels},
        output_dir / "cached_validation_features.pt",
    )

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=classifier_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels),
        batch_size=classifier_batch_size,
        shuffle=False,
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = []
    best_loss = float("inf")
    best_accuracy = -1.0
    best_accuracy_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _feature_epoch(
            model.head, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = _feature_epoch(
            model.head, validation_loader, loss_function, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_validation_loss.pth")
        if validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_accuracy_loss
        ):
            best_accuracy = validation_accuracy
            best_accuracy_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_validation_accuracy.pth")

    torch.save(model.state_dict(), output_dir / "final_epoch.pth")
    save_history(history, output_dir)

    # Use the image loader for the final evaluation and prediction filenames.
    summary = {}
    for name, filename in (
        ("final", "final_epoch.pth"),
        ("best_loss", "best_validation_loss.pth"),
        ("best_accuracy", "best_validation_accuracy.pth"),
    ):
        model.load_state_dict(torch.load(output_dir / filename, map_location=device, weights_only=True))
        summary[name] = save_evaluation(
            name, model, image_validation, loss_function, device, output_dir
        )

    configuration = {
        "phase": "frozen",
        "method": "fixed feature extraction followed by classifier training",
        "why_feature_cache_is_exact": "The backbone is frozen and no random augmentation is used, so its features do not change between epochs.",
        "epochs": epochs,
        "classifier_batch_size": classifier_batch_size,
        "image_extraction_batch_size": image_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "seed": 42,
        "training_images": 176,
        "validation_images": 45,
        "test_images": 0,
        "test_set_note": "No independent test set exists in the authoritative 176/45 split.",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    (output_dir / "run_configuration.json").write_text(json.dumps(configuration, indent=2))
    (output_dir / "result_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
