
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_datasets
from baseline import SimpleCNN


def calculate_metrics(labels, probabilities):
    predictions = [int(p >= 0.5) for p in probabilities]

    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0

    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    f1_1 = (
        2 * precision_1 * recall_1 / (precision_1 + recall_1)
        if precision_1 + recall_1 else 0.0
    )

    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    f1_0 = (
        2 * precision_0 * recall_0 / (precision_0 + recall_0)
        if precision_0 + recall_0 else 0.0
    )

    # AUROC: pairwise ranking with ties counting as 0.5
    positives = sum(labels)
    negatives = len(labels) - positives
    auc = None

    if positives > 0 and negatives > 0:
        ranked = sorted(zip(probabilities, labels))
        rank_sum = 0.0
        i = 0

        while i < len(ranked):
            j = i + 1
            while j < len(ranked) and ranked[j][0] == ranked[i][0]:
                j += 1

            average_rank = ((i + 1) + j) / 2.0
            rank_sum += average_rank * sum(
                label for _, label in ranked[i:j]
            )
            i = j

        auc = (
            rank_sum - positives * (positives + 1) / 2
        ) / (positives * negatives)

    return {
        "accuracy": accuracy,
        "precision_class_0": precision_0,
        "recall_class_0": recall_0,
        "precision_class_1": precision_1,
        "recall_class_1": recall_1,
        "macro_f1": (f1_0 + f1_1) / 2,
        "auroc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(model, loader, optimizer, criterion,
                    device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_number, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, max_batches=None):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probabilities = []

    for batch_number, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probabilities = torch.softmax(outputs, dim=1)[:, 1]

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    metrics = calculate_metrics(all_labels, all_probabilities)
    metrics["val_loss"] = total_loss / max(total_samples, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    train_dataset, val_dataset, test_dataset = load_datasets()

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    output_dir = Path(
        "smoke_outputs" if is_smoke_test else "checkpoints"
    )
    output_dir.mkdir(exist_ok=True)

    best_accuracy = -1.0
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.max_train_batches
        )

        metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            args.max_val_batches
        )

        accuracy = metrics["accuracy"]

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {metrics['val_loss']:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")

        if metrics["auroc"] is not None:
            print(f"AUROC: {metrics['auroc']:.4f}")
        else:
            print("AUROC: unavailable (only one class present)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                output_dir / "baseline_best.pth"
            )
            print("Best model saved.")

    with open(output_dir / "history.json", "w") as file:
        json.dump(history, file, indent=2)

    print("\nTraining finished.")
    print("Best validation accuracy:", best_accuracy)
    print("Results saved to:", output_dir)

    if is_smoke_test:
        print("DEVELOPMENT SMOKE TEST - not a final result.")


if __name__ == "__main__":
    main()
