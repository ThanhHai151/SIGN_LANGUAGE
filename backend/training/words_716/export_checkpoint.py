"""
Utility script to package, inspect, and export trained checkpoints and training logs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch


def inspect_and_export_checkpoint(
    checkpoint_dir: Path,
    export_dir: Path,
    checkpoint_name: str = "best.pth",
):
    ckpt_path = checkpoint_dir / checkpoint_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    export_dir.mkdir(parents=True, exist_ok=True)
    target_path = export_dir / checkpoint_name
    shutil.copy2(ckpt_path, target_path)

    # Compute SHA-256
    h = hashlib.sha256()
    with open(target_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha256 = h.hexdigest()

    ckpt = torch.load(target_path, map_location="cpu", weights_only=False)

    meta = {
        "export_time": datetime.now(timezone.utc).isoformat(),
        "checkpoint_filename": checkpoint_name,
        "checkpoint_sha256": sha256,
        "size_bytes": target_path.stat().st_size,
        "schema_version": ckpt.get("schema_version"),
        "epoch": ckpt.get("epoch"),
        "phase": ckpt.get("phase"),
        "loss": ckpt.get("loss"),
        "accuracy": ckpt.get("accuracy"),
        "num_classes": ckpt.get("num_classes"),
        "active_classes_count": len(ckpt.get("active_indices", [])),
    }

    report_path = checkpoint_dir / "training_report.json"
    if report_path.is_file():
        shutil.copy2(report_path, export_dir / "training_report.json")

    hist_path = checkpoint_dir / "history.csv"
    if hist_path.is_file():
        shutil.copy2(hist_path, export_dir / "history.csv")

    (export_dir / "export_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[EXPORT] Successfully exported {checkpoint_name} (SHA-256: {sha256[:12]}...) to {export_dir}")
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--export-dir", type=str, default="./exports")
    parser.add_argument("--name", type=str, default="best.pth")
    args = parser.parse_args()
    inspect_and_export_checkpoint(Path(args.ckpt_dir), Path(args.export_dir), args.name)
