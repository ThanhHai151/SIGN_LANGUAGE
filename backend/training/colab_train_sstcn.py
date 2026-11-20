"""Package a prepared unified v2 dataset for free Kaggle/Colab GPU training.

Because the local workstation has no NVIDIA GPU, this helper creates a compact
bundle zip (`sequences.npy`, `labels.npy`, `classes.json`, `topics.json`,
`manifest.json`) and prints the exact 3-line command to run on Kaggle / Google Colab
GPU with KVSL online augmentations, class-balanced loss, and cosine LR schedule.

Usage:
  python -m backend.training.colab_train_sstcn \
    --data backend/data/vsl_unified/prepared \
    --bundle backend/data/vsl_unified/vsl_unified_gpu_bundle.zip
"""

import argparse
import zipfile
from pathlib import Path

BUNDLE_FILES = ("sequences.npy", "labels.npy", "classes.json", "topics.json", "manifest.json", "harmonization_report.json")


def create_gpu_bundle(data_dir: Path, bundle_path: Path) -> Path:
    data_dir = Path(data_dir)
    bundle_path = Path(bundle_path)
    missing = [f for f in ("sequences.npy", "labels.npy", "classes.json", "manifest.json") if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required prepared dataset files in {data_dir}: {missing}")

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fname in BUNDLE_FILES:
            fpath = data_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"prepared/{fname}")
    return bundle_path


def print_gpu_instructions(bundle_path: Path):
    print(f"\n✅ Đã đóng gói bộ dữ liệu GPU tại: {bundle_path}")
    print("\n--- Hướng dẫn train trên Google Colab / Kaggle GPU ---")
    print("1) Upload file zip lên Google Drive hoặc Kaggle Dataset.")
    print("2) Trong notebook GPU (T4 / P100), chạy lệnh:")
    print(
        """
!git clone <REPO_URL> VSL && cd VSL && unzip -o /path/to/vsl_unified_gpu_bundle.zip -d backend/data/vsl_unified/
!python -m backend.training.train_sstcn \\
  --data backend/data/vsl_unified/prepared \\
  --output backend/runs/sstcn_unified_v2 \\
  --epochs 60 --batch-size 32 --lr 1e-3 --dropout 0.3 \\
  --augment --augment-profile kvsl \\
  --class-balanced --label-smoothing 0.1 --scheduler cosine \\
  --patience 10 --amp
""".strip()
    )
    print("\n3) Tải `backend/runs/sstcn_unified_v2/best.pth` về máy đặt tại:")
    print("   backend/models/weights/sign_sstcn_unified_v2.pth")
    print("4) Khởi chạy dự án chính với model mới:")
    print("   VSL_SSTCN_WEIGHTS=backend/models/weights/sign_sstcn_unified_v2.pth ./start.sh\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("backend/data/vsl_unified/prepared"))
    p.add_argument("--bundle", type=Path, default=Path("backend/data/vsl_unified/vsl_unified_gpu_bundle.zip"))
    args = p.parse_args()
    out = create_gpu_bundle(args.data, args.bundle)
    print_gpu_instructions(out)


if __name__ == "__main__":
    main()
