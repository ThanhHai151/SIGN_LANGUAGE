"""Train letters and digits from source-grouped video skeletons."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.models.fingerspelling_sequence import FingerspellingSequence, select_dominant_hand, sequence_features
from backend.training.sstcn_data import DATA_ROOT, load_prepared, write_json


def prepare_splits(data):
    x, y, manifest = load_prepared(data)
    classes = [value.removeprefix("Số ") for value in manifest["class_names"][:39]]
    splits = {}
    for split in ("train", "val", "test"):
        rows = [item["row"] for item in manifest["samples"] if item["split"] == split and item["label"] < 39]
        features = []
        for offset in range(0, len(rows), 512):
            features.append(sequence_features(select_dominant_hand(x[rows[offset:offset + 512]])))
        splits[split] = (torch.from_numpy(np.concatenate(features)), torch.from_numpy(np.asarray(y[rows], dtype=np.int64)))
    return classes, splits


def score(model, dataset, device):
    counts = np.zeros((39, 39), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for x, y in dataset:
            logits = model(x.to(device))
            split = y < 29
            predictions = torch.empty_like(y)
            predictions[split] = logits[split, :29].argmax(dim=1).cpu()
            predictions[~split] = logits[~split, 29:].argmax(dim=1).cpu() + 29
            np.add.at(counts, (y.numpy(), predictions.numpy()), 1)
    return counts


def stats(matrix):
    summary = {}
    for name, group in (("letters", slice(0, 29)), ("digits_1_to_10", slice(29, 39))):
        correct = int(matrix.diagonal()[group].sum())
        total = int(matrix[group].sum())
        summary[name] = {"correct": correct, "total": total, "accuracy": correct / total}
    return summary


def augment(data):
    x = data.clone()
    n = len(x)
    points = x[:, :63].reshape(n, 21, 3, -1)
    # Camera frames may arrive mirrored, rolled, or with different depth scale.
    mirror = torch.where(torch.rand((n, 1, 1), device=x.device) < 0.25, -1.0, 1.0)
    points[:, :, 0] *= mirror
    x[:, 63] *= mirror[:, 0]
    x[:, 66] *= mirror[:, 0]
    theta = torch.randn((n, 1, 1), device=x.device).clamp(-2.0, 2.0) * 0.085
    cos, sin = theta.cos(), theta.sin()
    p_x, p_y = points[:, :, 0].clone(), points[:, :, 1].clone()
    points[:, :, 0], points[:, :, 1] = p_x * cos - p_y * sin, p_x * sin + p_y * cos
    points[:, :, 2] *= 0.80 + 0.40 * torch.rand((n, 1, 1), device=x.device)
    x += torch.randn_like(x) * 0.018
    return x


def train(data, output, epochs, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(min(5, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, splits = prepare_splits(data)
    loaders = {
        name: DataLoader(TensorDataset(*arrays), batch_size=128, shuffle=name == "train")
        for name, arrays in splits.items()
    }
    model = FingerspellingSequence(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0011, weight_decay=0.0003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    best = -1
    for epoch in range(epochs):
        model.train()
        for x, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(augment(x.to(device))), y.to(device), label_smoothing=0.025)
            loss.backward()
            optimizer.step()
        schedule.step()
        matrix = score(model, loaders["val"], device)
        accuracy = float(matrix.trace() / matrix.sum())
        print(f"epoch {epoch + 1}/{epochs} val {stats(matrix)}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"classes": classes, "state_dict": model.state_dict(), "epoch": epoch + 1}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    matrix = score(model, loaders["test"], device)
    report = {
        "selected_epoch": checkpoint["epoch"], "class_names": classes,
        "validation": stats(score(model, loaders["val"], device)), "test": stats(matrix),
        "test_confusion_matrix": matrix.tolist(),
    }
    write_json(output / "metrics.json", report)
    print(report["test"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT / "prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.seed)
