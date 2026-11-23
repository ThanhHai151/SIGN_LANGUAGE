"""
Local Training and Fine-Tuning Pipeline for WordRecognizer716 on Local CPU.

Supports:
- Phase A: Frozen Pretrained Backbone -> train 716-class classification head.
- Phase B: Partial Fine-Tuning -> unfreeze temporal_conv2 with lower learning rate.
- Active class masking: supports Pilot 40 classes or Full 703 classes.
- Label smoothing confined strictly to active classes.
- BatchNorm running statistics preservation for frozen backbone.
- Full reproducibility with deterministic seeding.
- Dry-run verification mode (--dry-run).
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Handle local package imports
try:
    from backend.training.words_716.model import WordRecognizer716
    from backend.training.words_716.dataset import WordDataset
    from backend.training.words_716.loss import MaskedCrossEntropyLoss
except ImportError:
    # Fallback for standalone script execution
    from model import WordRecognizer716
    from dataset import WordDataset
    from loss import MaskedCrossEntropyLoss


def get_system_ram_gb() -> float:
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        used_gb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024**3)
        return round(used_gb, 2)
    except Exception:
        return 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_epoch(
    model: WordRecognizer716,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp: bool = False,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    num_batches = 0

    with torch.set_grad_enabled(is_training):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(inputs)
                loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss encountered during training!")

            grad_norm = 0.0
            if is_training:
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
                    ).item()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
                    ).item()
                    optimizer.step()

                total_grad_norm += grad_norm

            n = targets.size(0)
            total_loss += loss.item() * n
            # Preds strictly from masked logits or active logits
            masked_logits = model.apply_mask(logits)
            preds = masked_logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += n
            num_batches += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1) if is_training else 0.0

    gpu_mem = 0.0
    if device.type == "cuda":
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "grad_norm": avg_grad_norm,
        "gpu_memory_mb": round(gpu_mem, 2),
        "samples": total,
    }


def save_checkpoint(
    path: Path,
    model: WordRecognizer716,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    acc: float,
    config: dict,
    phase: str,
):
    tmp = path.with_suffix(".pth.tmp")
    payload = {
        "schema_version": "3.2a-words-716-checkpoint-v1",
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": acc,
        "num_classes": model.num_classes,
        "active_indices": model.active_indices,
        "unavailable_indices": model.unavailable_indices,
        "config": config,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def train_pipeline(args):
    # 1. Deterministic Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Device Configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    # 3. Load Dataset
    data_dir = Path(args.data_dir)
    seq_path = data_dir / "sequences.npy"
    lbl_path = data_dir / "labels.npy"
    failed_path = data_dir / "failed_samples.jsonl"
    pilot_path = Path(args.active_classes_file) if args.active_classes_file else None

    if not seq_path.is_file() or not lbl_path.is_file():
        raise FileNotFoundError(f"Missing sequences.npy or labels.npy in {data_dir}")

    sequences = np.load(seq_path, mmap_mode="r")
    labels = np.load(lbl_path)

    # Determine active and unavailable indices
    active_indices = None
    if pilot_path and pilot_path.is_file():
        pilot_data = json.loads(pilot_path.read_text(encoding="utf-8"))
        active_indices = pilot_data.get("pilot_class_indices") or [c["class_index"] for c in pilot_data.get("classes", [])]
        print(f"[PILOT] Loaded active classes from {pilot_path.name}: {len(active_indices)} classes.")
    elif failed_path.is_file():
        failed_rows = [json.loads(l) for l in failed_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        unavail = set(r["class_index"] for r in failed_rows)
        active_indices = [i for i in range(716) if i not in unavail]

    # Filter dataset rows if active_indices is specified (e.g. for Pilot 40)
    if active_indices is not None and len(active_indices) < len(labels):
        active_set = set(active_indices)
        selected_rows = [row_idx for row_idx, lbl in enumerate(labels) if int(lbl) in active_set]
    else:
        selected_rows = list(range(len(labels)))

    n_samples = len(selected_rows)
    print(f"[INIT] Active dataset samples: {n_samples} (from total {len(labels)}).")
    print(f"[INIT] Active classes count: {len(active_indices) if active_indices else 716}")
    print(f"[INIT] Target device: {device} (AMP: {amp})")

    # 4. Initialize Model & Criterion
    model = WordRecognizer716(
        num_classes=716,
        num_joints=75,
        in_channels=3,
        dropout=args.dropout,
        active_indices=active_indices,
    ).to(device)

    criterion = MaskedCrossEntropyLoss(
        num_classes=716,
        active_indices=active_indices,
        label_smoothing=args.label_smoothing,
    ).to(device)

    # 5. Load Pretrained Backbone
    weights_path = Path(args.pretrained_weights)
    transfer_info = {}
    if weights_path.is_file():
        transfer_info = model.load_pretrained_backbone(str(weights_path))
        print(f"[BACKBONE] Loaded {transfer_info['loaded_backbone_keys_count']} keys from {weights_path.name}")
    else:
        print(f"[BACKBONE] Warning: Pretrained weights file not found at {weights_path}; initializing randomly.")

    # 6. Dry-Run Verification Mode
    if args.dry_run:
        print("\n--- RUNNING DRY-RUN VERIFICATION PASS ---")
        model.eval()
        dummy_batch_size = min(4, n_samples)
        batch_indices = selected_rows[:dummy_batch_size]
        batch_seqs = np.array(sequences[batch_indices], copy=True)
        if args.body_centric:
            try:
                from backend.training.words_716.normalization import normalize_batch
            except ImportError:
                from normalization import normalize_batch
            batch_seqs = normalize_batch(batch_seqs)
        sample_x = torch.from_numpy(batch_seqs).to(device)
        sample_y = torch.from_numpy(np.array(labels[batch_indices], copy=True)).to(device)

        print(f"Dry-run input shape: {sample_x.shape}")
        out_logits = model(sample_x)
        masked_logits = model(sample_x, apply_class_mask=True)
        loss_val = criterion(out_logits, sample_y)

        # Verify gradient flow
        loss_val.backward()
        fc_has_grad = model.fc.weight.grad is not None and torch.any(model.fc.weight.grad != 0)
        print(f"Output logits shape: {out_logits.shape} (Dtype: {out_logits.dtype})")
        print(f"Masked logits for inactive: {masked_logits[:, model.unavailable_indices[:3]].max().item():.1e}")
        print(f"Loss computed: {loss_val.item():.4f}")
        print(f"Backward pass gradient flow verified: {fc_has_grad}")
        print("Model parameters summary:", model.count_parameters())
        print("--- DRY-RUN VERIFICATION COMPLETED SUCCESSFULLY ---\n")
        return {
            "status": "DRY_RUN_SUCCESS",
            "device": str(device),
            "input_shape": list(sample_x.shape),
            "output_shape": list(out_logits.shape),
            "loss": float(loss_val.item()),
            "parameters": model.count_parameters(),
            "transfer_info": transfer_info,
            "fc_has_grad": bool(fc_has_grad),
        }

    # 7. Setup Output Directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "phase": args.phase,
        "epochs_a": args.epochs_a,
        "epochs_b": args.epochs_b,
        "batch_size": args.batch_size,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "scale_range": [args.scale_min, args.scale_max],
        "device": str(device),
        "amp": amp,
        "body_centric_normalization": bool(args.body_centric),
        "data_samples": n_samples,
        "active_classes_count": len(model.active_indices),
        "unavailable_classes_count": len(model.unavailable_indices),
        "transfer_info": transfer_info,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Build Training Dataset & Loader
    train_dataset = WordDataset(
        sequences,
        labels,
        indices=selected_rows,
        augment=True,
        seed=args.seed,
        scale_range=(args.scale_min, args.scale_max),
        body_centric=args.body_centric,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    history = []
    best_loss = float("inf")

    total_epochs = 0
    if args.phase in ("A", "A_then_B"):
        total_epochs += args.epochs_a
    if args.phase in ("B", "A_then_B"):
        total_epochs += args.epochs_b
    train_start_time = time.time()

    # PHASE A: Frozen Backbone
    if args.phase in ("A", "A_then_B"):
        print("\n================== STARTING PHASE A: FROZEN BACKBONE ==================")
        model.freeze_backbone(freeze=True)
        print("Parameters:", model.count_parameters())

        optimizer_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )

        for ep in range(1, args.epochs_a + 1):
            t0 = time.time()
            train_dataset.set_epoch(ep)
            metrics = run_epoch(model, train_loader, criterion, device, optimizer_a, scaler, amp)
            elapsed = time.time() - t0

            ram_now = get_system_ram_gb()
            row = {
                "epoch": ep,
                "phase": "A",
                "loss": round(metrics["loss"], 4),
                "acc": round(metrics["accuracy"], 4),
                "lr": args.lr_head,
                "grad_norm": round(metrics["grad_norm"], 4),
                "gpu_mem_mb": metrics["gpu_memory_mb"],
                "ram_used_gb": ram_now,
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(row)

            step_overall = len(history)
            pct = (step_overall / total_epochs) * 100.0 if total_epochs > 0 else 0.0
            total_elapsed = time.time() - train_start_time
            avg_sec = total_elapsed / step_overall if step_overall > 0 else elapsed
            rem_sec = avg_sec * (total_epochs - step_overall)
            eta_str = f"{int(rem_sec // 60)}m {int(rem_sec % 60):02d}s" if rem_sec >= 0 else "0m 00s"

            is_best = metrics["loss"] < best_loss
            ckpt_tag = " -> [BEST SAVED]" if is_best else ""

            print(
                f"[{pct:5.1f}% | Ep {step_overall:02d}/{total_epochs:02d} | Phase A] "
                f"loss={row['loss']:.4f} acc={row['acc']*100:5.2f}% | "
                f"RAM={ram_now:.1f}GB | time={elapsed:.1f}s (ETA: {eta_str}){ckpt_tag}"
            )

            if is_best:
                best_loss = metrics["loss"]
                save_checkpoint(out_dir / "best.pth", model, optimizer_a, ep, metrics["loss"], metrics["accuracy"], run_config, "A")
            save_checkpoint(out_dir / "last.pth", model, optimizer_a, ep, metrics["loss"], metrics["accuracy"], run_config, "A")

        save_checkpoint(out_dir / "phase_a_final.pth", model, optimizer_a, args.epochs_a, history[-1]["loss"], history[-1]["acc"], run_config, "A")
        print(f"[Phase A Complete] Saved {out_dir / 'phase_a_final.pth'}")

    # PHASE B: Partial Fine-Tuning
    if args.phase in ("B", "A_then_B"):
        print("\n================== STARTING PHASE B: PARTIAL FINE-TUNING ==================")
        model.unfreeze_partial_backbone(unfreeze_temporal2=True)
        print("Parameters:", model.count_parameters())

        param_groups = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "fc" in n], "lr": args.lr_head * 0.5},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "fc" not in n], "lr": args.lr_backbone},
        ]
        optimizer_b = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        start_ep = args.epochs_a + 1 if args.phase == "A_then_B" else 1
        end_ep = start_ep + args.epochs_b - 1

        for ep in range(start_ep, end_ep + 1):
            t0 = time.time()
            train_dataset.set_epoch(ep)
            metrics = run_epoch(model, train_loader, criterion, device, optimizer_b, scaler, amp)
            elapsed = time.time() - t0

            ram_now = get_system_ram_gb()
            row = {
                "epoch": ep,
                "phase": "B",
                "loss": round(metrics["loss"], 4),
                "acc": round(metrics["accuracy"], 4),
                "lr": args.lr_backbone,
                "grad_norm": round(metrics["grad_norm"], 4),
                "gpu_mem_mb": metrics["gpu_memory_mb"],
                "ram_used_gb": ram_now,
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(row)

            step_overall = len(history)
            pct = (step_overall / total_epochs) * 100.0 if total_epochs > 0 else 0.0
            total_elapsed = time.time() - train_start_time
            avg_sec = total_elapsed / step_overall if step_overall > 0 else elapsed
            rem_sec = avg_sec * (total_epochs - step_overall)
            eta_str = f"{int(rem_sec // 60)}m {int(rem_sec % 60):02d}s" if rem_sec >= 0 else "0m 00s"

            is_best = metrics["loss"] < best_loss
            ckpt_tag = " -> [BEST SAVED]" if is_best else ""

            print(
                f"[{pct:5.1f}% | Ep {step_overall:02d}/{total_epochs:02d} | Phase B] "
                f"loss={row['loss']:.4f} acc={row['acc']*100:5.2f}% | "
                f"RAM={ram_now:.1f}GB | time={elapsed:.1f}s (ETA: {eta_str}){ckpt_tag}"
            )

            if is_best:
                best_loss = metrics["loss"]
                save_checkpoint(out_dir / "best.pth", model, optimizer_b, ep, metrics["loss"], metrics["accuracy"], run_config, "B")
            save_checkpoint(out_dir / "last.pth", model, optimizer_b, ep, metrics["loss"], metrics["accuracy"], run_config, "B")

        save_checkpoint(out_dir / "phase_b_final.pth", model, optimizer_b, end_ep, history[-1]["loss"], history[-1]["acc"], run_config, "B")
        print(f"[Phase B Complete] Saved {out_dir / 'phase_b_final.pth'}")

    # Copy pilot_40_classes.json if used
    if pilot_path and pilot_path.is_file():
        import shutil
        shutil.copy2(pilot_path, out_dir / "pilot_40_classes.json")

    # Export History CSV
    if history:
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # Export Training Report JSON
    report = {
        "schema_version": "3.2a-pilot-training-report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "best_loss": best_loss,
        "final_accuracy": history[-1]["acc"] if history else None,
        "epochs_completed": len(history),
        "history": history,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[DONE] Training pipeline completed. Checkpoints stored in: {out_dir}")
    return {"status": "SUCCESS", "history": history, "best_loss": best_loss}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WordRecognizer716 Fine-Tuning Pipeline")
    p.add_argument("--data-dir", type=str, default="backend/data/words_716", help="Path to words_716 dataset")
    p.add_argument("--pretrained-weights", type=str, default="backend/models/weights/sign_sstcn_attention_model.pth", help="Pretrained SSTCN weights")
    p.add_argument("--active-classes-file", type=str, default=None, help="Path to JSON file specifying active classes (e.g. pilot_40_classes.json)")
    p.add_argument("--output-dir", type=str, default="backend/models/words_716_checkpoints", help="Output directory")
    p.add_argument("--phase", type=str, default="A_then_B", choices=["A", "B", "A_then_B"])
    p.add_argument("--epochs-a", type=int, default=15, help="Epochs for Phase A")
    p.add_argument("--epochs-b", type=int, default=20, help="Epochs for Phase B")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--scale-min", type=float, default=0.95, help="Min scale augmentation factor (default: 0.95)")
    p.add_argument("--scale-max", type=float, default=1.05, help="Max scale augmentation factor (default: 1.05)")
    p.add_argument("--body-centric", action="store_true", help="Enable Body-Centric Normalization in WordDataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    p.add_argument("--dry-run", action="store_true", help="Run 1 forward and loss pass without weight updates")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train_pipeline(args)
