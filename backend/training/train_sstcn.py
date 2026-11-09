"""Train the deployed 102-class SSTCN architecture with grouped splits and audit artifacts."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.models.sstcn_model import SSTCN_Attention
from backend.training.evaluate_sstcn import SequenceDataset, evaluate
from backend.training.fetch_sstcn_data import sha256
from backend.training.sstcn_data import CLASS_NAMES, DATA_ROOT, PREPROCESSING, class_mapping, load_prepared, write_json


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; no checkpoint was promoted")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            n = len(targets)
            total_loss += loss.item() * n
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += n
    return {"loss": total_loss / total, "accuracy": correct / total, "samples": total}


def save_checkpoint(path, value):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write_history(directory, history):
    with (directory / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{key}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=key, title=f"SSTCN {key}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(fig)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0 or args.threads < 1:
        raise ValueError("epochs, batch-size, patience, lr and threads must be positive")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0:
        raise ValueError("dropout must be in [0,1), weight-decay must be non-negative")
    if args.resume and args.init_from:
        raise ValueError("Choose --resume or --init-from")
    out = args.output
    if args.resume and (args.resume.resolve() != (out / "last.pth").resolve()
                        or not (out / "best.pth").exists()):
        raise ValueError("Resume from last.pth into its existing run directory containing best.pth")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory {out} is not empty; use a new --output or --resume")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    x, y, manifest = load_prepared(args.data)
    ids = {s: [r["row"] for r in manifest["samples"] if r["split"] == s] for s in ("train", "val", "test")}
    for split in ("train", "val"):
        if not ids[split]:
            raise ValueError(f"Missing {split} split")
        missing = set(range(102)) - {int(y[i]) for i in ids[split]}
        if missing and not args.allow_partial_classes:
            raise ValueError(f"{split} lacks classes {sorted(missing)}; --allow-partial-classes is for development only")
    train_hashes = {manifest["samples"][i]["sha256"] for i in ids["train"]}
    if train_hashes & {manifest["samples"][i]["sha256"] for i in ids["val"] + ids["test"]}:
        raise ValueError("Exact sequence overlap between training and held-out data")
    train_groups = {(r["label"], r["group"]) for r in manifest["samples"] if r["split"] == "train"}
    val_groups = {(r["label"], r["group"]) for r in manifest["samples"] if r["split"] == "val"}
    if train_groups & val_groups:
        raise ValueError("Source-video groups overlap between training and validation")
    model = SSTCN_Attention(num_classes=102, dropout=args.dropout).to(device)
    if args.init_from:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=True)
        if initial.get("class_names", CLASS_NAMES) != CLASS_NAMES:
            raise ValueError("Initial checkpoint class mapping differs")
        model.load_state_dict(initial.get("model_state_dict", initial), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    digest = sha256(args.data / "manifest.json")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"created_at": datetime.now(timezone.utc).isoformat(), "torch_version": str(torch.__version__),
                   "numpy_version": str(np.__version__), "device": str(device), "dataset_manifest_sha256": digest,
                   "python_version": sys.version, "cuda_version": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "initial_checkpoint_sha256": sha256(args.init_from) if args.init_from else None,
                   "counts": {s: len(i) for s, i in ids.items()},
                   "selection": "validation accuracy, then validation loss; test is evaluated only after selection",
                   "historical_initialization_training_membership": "unverified" if args.init_from else "not_applicable"})
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    config["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    config["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    code_root = Path(__file__).resolve().parents[1]
    config["source_sha256"] = {str(p.relative_to(code_root)): sha256(p)
        for p in [Path(__file__).resolve(), code_root / "models/sstcn_model.py",
                  code_root / "training/evaluate_sstcn.py", code_root / "training/sstcn_data.py",
                  code_root / "training/prepare_sstcn.py"]}
    history, start, best_score, bad_epochs = [], 0, (-1.0, float("-inf")), 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ckpt.get("dataset_manifest_sha256") != digest or ckpt.get("class_names") != CLASS_NAMES:
            raise ValueError("Resume requires the same prepared dataset and mapping")
        for key in ("seed", "batch_size", "dropout", "lr", "weight_decay", "amp", "augment"):
            if ckpt["run_config"][key] != config[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        if device.type == "cuda" and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        history, start = ckpt["history"], ckpt["epoch"] + 1
        best_score, bad_epochs = tuple(ckpt["best_score"]), ckpt["bad_epochs"]
        # Preserve the initialization lineage across resumed runs.
        config["initial_checkpoint_sha256"] = ckpt["run_config"].get("initial_checkpoint_sha256")
        config["historical_initialization_training_membership"] = ckpt["run_config"].get("historical_initialization_training_membership")
    if start >= args.epochs:
        raise ValueError("--epochs is the total epoch count and must exceed the resumed epoch")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", config)
    write_json(out / "classes.json", class_mapping())
    shutil.copyfile(args.data / "manifest.json", out / "dataset_manifest.json")
    train_ds = SequenceDataset(x, y, ids["train"], args.augment, args.seed)
    val_loader = DataLoader(SequenceDataset(x, y, ids["val"]), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    print(f"Training on {device}: {config['counts']}, amp={amp}", flush=True)
    for epoch in range(start, args.epochs):
        started = time.monotonic()
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        train_ds.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator,
                            num_workers=0, pin_memory=device.type == "cuda")
        train_result = run_epoch(model, loader, device, optimizer, scaler, amp)
        val_result = run_epoch(model, val_loader, device, amp=amp)
        scheduler.step(val_result["loss"])
        score = (val_result["accuracy"], -val_result["loss"])
        improved = score > best_score
        bad_epochs = 0 if improved else bad_epochs + 1
        if improved:
            best_score = score
        record = {"epoch": epoch, "train_loss": train_result["loss"], "train_accuracy": train_result["accuracy"],
                  "val_loss": val_result["loss"], "val_accuracy": val_result["accuracy"],
                  "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started}
        history.append(record)
        checkpoint = {"schema_version": 1, "epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                      "scaler_state_dict": scaler.state_dict(), "loss": val_result["loss"],
                      "class_names": CLASS_NAMES, "preprocessing": PREPROCESSING,
                      "model_config": {"num_classes": 102, "dropout": args.dropout},
                      "dataset_manifest_sha256": digest, "run_config": config,
                      "history": history, "best_score": list(best_score), "bad_epochs": bad_epochs,
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(out / "last.pth", checkpoint)
        if improved:
            save_checkpoint(out / "best.pth", checkpoint)
        write_history(out, history)
        print(json.dumps(record), flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without validation improvement", flush=True)
            break
    if not (out / "best.pth").exists():
        raise FileNotFoundError("The best checkpoint is missing; resume into its original run directory")
    evaluate(out / "best.pth", args.data, out / "validation", "val", args.batch_size, str(device))
    if ids["test"]:
        evaluate(out / "best.pth", args.data, out / "test", "test", args.batch_size, str(device))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_ROOT / "prepared")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    p.add_argument("--augment", action="store_true", help="Optional online augmentation; upstream already contains augmented samples")
    p.add_argument("--init-from", type=Path, help="Load model weights only, starting a new optimizer/run")
    p.add_argument("--resume", type=Path, help="Continue a new-format last.pth in the same run directory")
    p.add_argument("--allow-partial-classes", action="store_true", help="Development only: permit a subset of classes")
    train(p.parse_args())


if __name__ == "__main__":
    main()
