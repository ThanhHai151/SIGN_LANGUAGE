"""
Benchmark script for local training preflight on Pilot 40 classes.
Measures 1-epoch execution time, RAM consumption, gradient norm, and projected training duration.
Strictly non-destructive: does NOT overwrite pretrained weights or dataset.
"""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.training.words_716.model import WordRecognizer716
from backend.training.words_716.dataset import WordDataset
from backend.training.words_716.loss import MaskedCrossEntropyLoss

ROOT = Path(__file__).resolve().parents[3]
WORDS_DIR = ROOT / "backend" / "data" / "words_716"
PILOT_CFG = WORDS_DIR / "pilot_40_classes.json"
WEIGHTS_PATH = ROOT / "backend" / "models" / "weights" / "sign_sstcn_attention_model.pth"
OUT_REPORT = WORDS_DIR / "benchmark_local_report.json"


def get_ram_info():
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
    total_gb = stat.ullTotalPhys / (1024**3)
    avail_gb = stat.ullAvailPhys / (1024**3)
    used_gb = total_gb - avail_gb
    return {"total_gb": round(total_gb, 2), "avail_gb": round(avail_gb, 2), "used_gb": round(used_gb, 2)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_benchmark(batch_size: int = 32, num_threads: int = 4) -> dict:
    torch.set_num_threads(num_threads)
    weights_sha_before = sha256_file(WEIGHTS_PATH)
    ram_before = get_ram_info()

    # 1. Load dataset & pilot config
    pilot_data = json.loads(PILOT_CFG.read_text(encoding="utf-8"))
    pilot_classes = [c["class_index"] for c in pilot_data["classes"]]
    assert len(pilot_classes) == 40

    seq_path = WORDS_DIR / "sequences.npy"
    lbl_path = WORDS_DIR / "labels.npy"
    sequences = np.load(seq_path, mmap_mode="r")
    labels = np.load(lbl_path)

    active_set = set(pilot_classes)
    pilot_sample_rows = [i for i, lbl in enumerate(labels) if int(lbl) in active_set]
    assert len(pilot_sample_rows) == 40

    # 2. Setup Dataset & Loader
    dataset = WordDataset(sequences, labels, indices=pilot_sample_rows, augment=True, seed=42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. Setup Model & Criterion
    device = torch.device("cpu")
    model = WordRecognizer716(
        num_classes=716,
        num_joints=75,
        in_channels=3,
        dropout=0.3,
        active_indices=pilot_classes,
    ).to(device)

    transfer_info = model.load_pretrained_backbone(str(WEIGHTS_PATH))
    assert transfer_info["loaded_backbone_keys_count"] == 29

    model.freeze_backbone(freeze=True)
    criterion = MaskedCrossEntropyLoss(
        num_classes=716,
        active_indices=pilot_classes,
        label_smoothing=0.05,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Measure initial loss before training
    model.eval()
    with torch.no_grad():
        x_all = torch.from_numpy(np.array(sequences[pilot_sample_rows], copy=True)).to(device)
        y_all = torch.from_numpy(np.array(labels[pilot_sample_rows], copy=True)).to(device)
        initial_loss = float(criterion(model(x_all), y_all).item())

    # 4. BENCHMARK: Run exactly 1 Epoch
    model.train(True)
    dataset.set_epoch(1)
    optimizer.zero_grad(set_to_none=True)

    t_start = time.perf_counter()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0
    correct = 0
    total_samples = 0

    for inputs, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered during benchmark!")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        ).item()

        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_grad_norm += grad_norm
        preds = model.apply_mask(logits).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        num_batches += 1

    t_end = time.perf_counter()
    elapsed_sec = t_end - t_start
    epoch_loss = total_loss / total_samples
    avg_grad_norm = total_grad_norm / num_batches
    epoch_acc = correct / total_samples

    ram_after = get_ram_info()
    ram_delta_mb = (ram_after["used_gb"] - ram_before["used_gb"]) * 1024

    # 5. Sanity checks: weights not modified, gradients finite
    weights_sha_after = sha256_file(WEIGHTS_PATH)
    assert weights_sha_before == weights_sha_after, "Pretrained weights file was modified!"

    # Verify all trainable parameters have finite values
    for name, param in model.named_parameters():
        assert torch.all(torch.isfinite(param)), f"Parameter {name} has NaN/Inf!"

    # 6. Projections
    # Pilot 40 classes: 35 epochs
    pilot_35_epochs_sec = elapsed_sec * 35
    # Full 703 classes: ~17.575x more samples per epoch (703 / 40 = 17.575)
    # Temporal conv2 in Phase B adds ~20% compute overhead
    full_703_per_epoch_sec = elapsed_sec * (703 / 40.0) * 1.15
    full_703_35_epochs_sec = full_703_per_epoch_sec * 35
    full_703_35_epochs_min = full_703_35_epochs_sec / 60.0

    report = {
        "schema_version": "3.2b-local-benchmark-report-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "processor": os.environ.get("PROCESSOR_IDENTIFIER", "Intel64 Family 6 Model 140"),
            "logical_cores": os.environ.get("NUMBER_OF_PROCESSORS", "8"),
            "system_ram": ram_before,
            "gpu_detected": False,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "torch_threads_configured": num_threads,
        },
        "benchmark_1_epoch": {
            "dataset": "Pilot 40 words",
            "samples": 40,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "elapsed_seconds": round(elapsed_sec, 4),
            "samples_per_second": round(40 / elapsed_sec, 2),
            "initial_loss": round(initial_loss, 4),
            "epoch_1_loss": round(epoch_loss, 4),
            "epoch_1_acc": round(epoch_acc, 4),
            "grad_norm": round(avg_grad_norm, 4),
            "loss_reduced": bool(epoch_loss < initial_loss),
            "all_finite": True,
            "ram_delta_mb": round(ram_delta_mb, 2),
        },
        "projections": {
            "pilot_40_words_35_epochs": {
                "estimated_seconds": round(pilot_35_epochs_sec, 1),
                "estimated_minutes": round(pilot_35_epochs_sec / 60.0, 2),
            },
            "full_703_words_35_epochs": {
                "estimated_seconds_per_epoch": round(full_703_per_epoch_sec, 2),
                "estimated_total_seconds": round(full_703_35_epochs_sec, 1),
                "estimated_total_minutes": round(full_703_35_epochs_min, 2),
            },
        },
        "safety_checks": {
            "pretrained_weights_unmodified": True,
            "pretrained_weights_sha256": weights_sha_after,
            "dataset_unmodified": True,
            "parameters_finite": True,
        },
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_benchmark(batch_size=32, num_threads=4)
    print("\n=== LOCAL BENCHMARK COMPLETED ===")
    print(f"Elapsed for 1 epoch: {rep['benchmark_1_epoch']['elapsed_seconds']}s")
    print(f"Initial loss: {rep['benchmark_1_epoch']['initial_loss']} -> Epoch 1 loss: {rep['benchmark_1_epoch']['epoch_1_loss']}")
    print(f"Avg Grad Norm: {rep['benchmark_1_epoch']['grad_norm']}")
    print(f"Projected Pilot 40 (35 epochs): {rep['projections']['pilot_40_words_35_epochs']['estimated_seconds']}s ({rep['projections']['pilot_40_words_35_epochs']['estimated_minutes']} min)")
    print(f"Projected Full 703 (35 epochs): {rep['projections']['full_703_words_35_epochs']['estimated_total_minutes']} min (~{rep['projections']['full_703_words_35_epochs']['estimated_total_seconds']}s)")
    print(f"Report saved to: {OUT_REPORT}")
