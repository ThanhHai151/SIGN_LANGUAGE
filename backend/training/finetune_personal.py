"""CPU-friendly Personal Calibration Fine-Tuner for SSTCN_Attention.

Adapts an existing SSTCN checkpoint (102-class v1 or unified multi-source v2)
to the user's own camera angle, body proportions, and signing style using
5-10 webcam samples per sign stored in `backend/data/personal_calibration/`.

Also supports adding brand-new words dynamically by expanding the final `fc`
layer while preserving existing weights.

Usage:
  python -m backend.training.finetune_personal \
    --base-checkpoint backend/models/weights/sign_sstcn_reconstructed_model.pth \
    --output backend/models/weights/sign_sstcn_personal.pth \
    --epochs 15
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from backend.models.sstcn_model import SSTCN_Attention, load_sstcn_model
from backend.training.augmentations import apply_kvsl_augmentations
from backend.training.label_harmonizer import LabelHarmonizer
from backend.training.sstcn_data import PREPROCESSING, canonical_sequence
from backend.vision.skeleton_norm import PREPROCESSING_V2, normalize_sequence_ctj_v2


class CalibrationDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int], prep_ver: int, repeat: int = 8, seed: int = 42):
        self.sequences = sequences
        self.labels = labels
        self.prep_ver = prep_ver
        self.repeat = max(1, repeat)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.sequences) * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % len(self.sequences)
        rep_id = idx // len(self.sequences)
        seq = self.sequences[base_idx].copy()
        if rep_id > 0:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, idx]))
            seq = apply_kvsl_augmentations(seq, rng, preprocessing_version=self.prep_ver)
        return torch.from_numpy(np.ascontiguousarray(seq)), int(self.labels[base_idx])


def _load_personal_samples(cal_dir: Path, prep_ver: int):
    samples = []
    if not cal_dir.exists():
        return samples
    for npy_path in sorted(cal_dir.rglob("*.npy")):
        meta_path = npy_path.with_suffix(".json")
        label = npy_path.parent.name
        entry_id = None
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                label = m.get("label") or label
                entry_id = m.get("entry_id")
            except Exception:
                pass
        raw = np.load(npy_path, allow_pickle=False)
        seq = canonical_sequence(raw)
        if prep_ver >= 2:
            seq = normalize_sequence_ctj_v2(seq)
        samples.append({"seq": seq, "label": label, "entry_id": entry_id, "path": str(npy_path)})
    return samples


def finetune_personal(
    base_checkpoint: Path,
    personal_dir: Path,
    output_path: Path,
    replay_data: Optional[Path] = None,
    replay_per_class: int = 4,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    repeat_factor: int = 8,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = load_sstcn_model(weights_path=str(base_checkpoint), device="cpu")
    prep_ver = int(getattr(base_model, "preprocessing_version", 1))
    class_names = list(base_model.class_names)
    entry_ids = list(base_model.entry_ids)
    topic_map = {k: list(v) for k, v in (base_model.topic_map or {}).items()}
    word_indices = list(base_model.word_class_indices)

    personal = _load_personal_samples(personal_dir, prep_ver=prep_ver)
    if not personal:
        raise ValueError(f"Không tìm thấy mẫu Personal Calibration nào trong {personal_dir}")

    harmonizer = LabelHarmonizer(split_regions=False)
    name_to_idx: Dict[str, int] = {n.strip().lower(): i for i, n in enumerate(class_names)}

    train_seqs: List[np.ndarray] = []
    train_labels: List[int] = []
    added_classes: List[str] = []

    for item in personal:
        raw_lbl = item["label"].strip()
        h = harmonizer.resolve(raw_lbl)
        cand_idx = name_to_idx.get(raw_lbl.lower())
        if cand_idx is None:
            cand_idx = name_to_idx.get(h["canonical_label"].lower())
        if cand_idx is None:
            cand_idx = len(class_names)
            class_names.append(h["canonical_label"])
            entry_ids.append(item["entry_id"] or h["entry_id"])
            word_indices.append(cand_idx)
            topic_map.setdefault(h["topic"], []).append(cand_idx)
            name_to_idx[h["canonical_label"].lower()] = cand_idx
            added_classes.append(h["canonical_label"])
        train_seqs.append(item["seq"])
        train_labels.append(cand_idx)

    # Optional replay samples from base dataset to reduce forgetting
    replay_count = 0
    if replay_data and (Path(replay_data) / "sequences.npy").exists():
        rx = np.load(Path(replay_data) / "sequences.npy", mmap_mode="r")
        ry = np.load(Path(replay_data) / "labels.npy", mmap_mode="r")
        rng = np.random.default_rng(seed)
        per_cls: Dict[int, List[int]] = {}
        for i, lbl in enumerate(ry):
            per_cls.setdefault(int(lbl), []).append(i)
        for lbl, idxs in per_cls.items():
            if lbl >= base_model.num_classes:
                continue
            chosen = rng.choice(idxs, size=min(replay_per_class, len(idxs)), replace=False)
            for c in chosen:
                seq_r = np.array(rx[int(c)], dtype=np.float32, copy=True)
                train_seqs.append(seq_r)
                train_labels.append(lbl)
                replay_count += 1

    new_num_classes = len(class_names)
    model = SSTCN_Attention(num_classes=new_num_classes, dropout=0.25)
    old_sd = base_model.state_dict()
    if new_num_classes == base_model.num_classes:
        model.load_state_dict(old_sd, strict=True)
    else:
        backbone_sd = {k: v for k, v in old_sd.items() if not k.startswith("fc.")}
        model.load_state_dict(backbone_sd, strict=False)
        with torch.no_grad():
            model.fc.weight[: base_model.num_classes].copy_(old_sd["fc.weight"])
            model.fc.bias[: base_model.num_classes].copy_(old_sd["fc.bias"])

    # Freeze early spatial/attention/temporal1 layers; fine-tune temporal_conv2 + fc
    for name, param in model.named_parameters():
        trainable = name.startswith(("temporal_conv2", "temporal_bn2", "fc"))
        param.requires_grad = trainable

    model = model.to(device)
    ds = CalibrationDataset(train_seqs, train_labels, prep_ver=prep_ver, repeat=repeat_factor, seed=seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    last_loss = 0.0
    last_acc = 0.0
    for ep in range(epochs):
        ds.epoch = ep
        model.train()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = torch.nn.functional.cross_entropy(logits, by, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(by)
            total_loss += float(loss.item()) * n
            correct += int((logits.argmax(dim=1) == by).sum().item())
            total += n
        last_loss = total_loss / max(1, total)
        last_acc = correct / max(1, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "schema_version": 2,
        "epoch": epochs,
        "loss": last_loss,
        "model_state_dict": model.cpu().state_dict(),
        "class_names": class_names,
        "entry_ids": entry_ids,
        "topic_map": topic_map,
        "word_class_indices": word_indices,
        "preprocessing": PREPROCESSING_V2 if prep_ver >= 2 else PREPROCESSING,
        "model_config": {"num_classes": new_num_classes, "dropout": 0.25},
        "personal_calibration": {
            "base_checkpoint": str(base_checkpoint),
            "personal_samples": len(personal),
            "replay_samples": replay_count,
            "added_classes": added_classes,
            "train_accuracy": round(last_acc, 4),
        },
    }
    tmp = output_path.with_name(output_path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(output_path)

    return {
        "output": str(output_path),
        "num_classes": new_num_classes,
        "preprocessing_version": prep_ver,
        "personal_samples": len(personal),
        "replay_samples": replay_count,
        "added_classes": added_classes,
        "final_train_loss": round(last_loss, 4),
        "final_train_accuracy": round(last_acc, 4),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, default=Path("backend/models/weights/sign_sstcn_reconstructed_model.pth"))
    p.add_argument("--personal-dir", type=Path, default=Path("backend/data/personal_calibration"))
    p.add_argument("--output", type=Path, default=Path("backend/models/weights/sign_sstcn_personal.pth"))
    p.add_argument("--replay-data", type=Path, default=None, help="Thư mục prepared gốc để replay chống quên")
    p.add_argument("--replay-per-class", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    res = finetune_personal(
        base_checkpoint=args.base_checkpoint,
        personal_dir=args.personal_dir,
        output_path=args.output,
        replay_data=args.replay_data,
        replay_per_class=args.replay_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        repeat_factor=args.repeat,
        seed=args.seed,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
