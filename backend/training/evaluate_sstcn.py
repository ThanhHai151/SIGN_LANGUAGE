"""Evaluate an SSTCN checkpoint and export reproducible metrics and predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from backend.models.sstcn_model import SSTCN_Attention
from backend.training.fetch_sstcn_data import sha256
from backend.training.sstcn_data import CLASS_NAMES, DATA_ROOT, PREPROCESSING, load_prepared, write_json


class SequenceDataset(Dataset):
    def __init__(self, x, y, indices, augment=False, seed=42):
        self.x, self.y, self.indices = x, y, list(indices)
        self.augment, self.seed, self.epoch = augment, seed, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        a = np.array(self.x[row], copy=True)
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, row]))
            present = np.any(a != 0, axis=0)
            center = np.array([0.5, 0.5, 0], dtype=np.float32)[:, None, None]
            a -= center
            angle = rng.uniform(-0.12, 0.12)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
            a = np.einsum("ij,jtv->itv", rotation, a) * rng.uniform(0.95, 1.05) + center
            a += rng.normal(0, 0.003, a.shape).astype(np.float32)
            a[:, ~present] = 0
            shift = rng.integers(-2, 3)
            a = a[:, np.clip(np.arange(30) + shift, 0, 29), :]
        return torch.from_numpy(np.ascontiguousarray(a)), int(self.y[row])


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = np.diag(matrix)
    precision = np.divide(tp, predicted, out=np.zeros(102, dtype=float), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(102, dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(102), where=(precision + recall) != 0)
    n = int(support.sum())
    if not n:
        raise ValueError("Cannot evaluate an empty split")
    return {"samples": n, "accuracy": float(tp.sum() / n),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "weighted_precision": float(precision @ support / n),
            "weighted_recall": float(recall @ support / n), "weighted_f1": float(f1 @ support / n),
            "represented_classes": int((support > 0).sum()),
            "per_class": [{"index": i, "label": CLASS_NAMES[i], "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                          for i in range(102)]}


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "class_names" in checkpoint and checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint class mapping differs from the deployed 102-class mapping")
    if "preprocessing" in checkpoint and checkpoint["preprocessing"] != PREPROCESSING:
        raise ValueError("Checkpoint preprocessing differs from the dataset contract")
    config = checkpoint.get("model_config", {})
    model = SSTCN_Attention(num_classes=102, dropout=config.get("dropout", 0.3))
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    return model.to(device).eval(), checkpoint


def evaluate(checkpoint_path, data_dir, output_dir, split="test", batch_size=8, device="auto"):
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    x, y, manifest = load_prepared(data_dir)
    indices = [r["row"] for r in manifest["samples"] if r["split"] == split]
    if not indices:
        raise ValueError(f"No samples in split {split}")
    model, ckpt = load_checkpoint_model(checkpoint_path, device)
    loader = DataLoader(SequenceDataset(x, y, indices), batch_size=batch_size, shuffle=False)
    confusion = np.zeros((102, 102), dtype=np.int64)
    total_loss, top5_correct, predictions = 0.0, 0, []
    with torch.inference_mode():
        offset = 0
        for inputs, target in loader:
            logits = model(inputs.to(device))
            total_loss += torch.nn.functional.cross_entropy(logits, target.to(device), reduction="sum").item()
            probabilities, top5 = logits.softmax(dim=1).topk(5, dim=1)
            probabilities, top5 = probabilities.cpu().numpy(), top5.cpu().numpy()
            truth = target.numpy()
            confusion += np.bincount(truth * 102 + top5[:, 0], minlength=102 * 102).reshape(102, 102)
            top5_correct += int((top5 == truth[:, None]).any(axis=1).sum())
            for j, label in enumerate(truth):
                r = manifest["samples"][indices[offset + j]]
                predictions.append({"source": r["source"], "sha256": r["sha256"],
                                    "true_index": int(label), "true_label": CLASS_NAMES[label],
                                    "predicted_index": int(top5[j, 0]), "predicted_label": CLASS_NAMES[top5[j, 0]],
                                    "confidence": float(probabilities[j, 0]),
                                    "top5_indices": json.dumps(top5[j].tolist())})
            offset += len(truth)
    metrics = metrics_from_confusion(confusion)
    metrics.update({"loss": total_loss / len(indices), "top5_accuracy": top5_correct / len(indices),
                    "split": split, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                    "dataset_manifest_sha256": sha256(Path(data_dir) / "manifest.json"),
                    "checkpoint_dataset_manifest_sha256": ckpt.get("dataset_manifest_sha256"),
                    "historical_checkpoint_training_membership": "unverified" if "dataset_manifest_sha256" not in ckpt else "recorded_in_checkpoint",
                    "confusion_axes": "rows=true label, columns=predicted label",
                    "macro_average": "all 102 model classes, zero for undefined metrics"})
    if not np.isfinite(total_loss):
        raise ValueError("Evaluation produced non-finite loss")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    np.save(output_dir / "confusion_matrix.npy", confusion, allow_pickle=False)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + CLASS_NAMES)
        writer.writerows([[CLASS_NAMES[i]] + row.tolist() for i, row in enumerate(confusion)])
    for filename, rows in (("per_class.csv", metrics["per_class"]), ("predictions.csv", predictions)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_confusion(confusion, output_dir / "confusion_matrix.png")
    print(json.dumps({k: metrics[k] for k in ("split", "samples", "accuracy", "top5_accuracy", "weighted_f1")}), flush=True)
    return metrics


def plot_confusion(confusion, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 16))
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0)
    plot = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(102), yticks=range(102), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted class", ylabel="True class", title="SSTCN confusion matrix (row normalized)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    fig.colorbar(plot, ax=ax, label="Fraction of true-class samples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA_ROOT / "prepared")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch size must be positive")
    evaluate(a.checkpoint, a.data, a.output, a.split, a.batch_size, a.device)


if __name__ == "__main__":
    main()
