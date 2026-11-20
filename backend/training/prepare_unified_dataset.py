"""Build a unified, leakage-free multi-source VSL dataset ready for `train_sstcn.py`.

Combines any subset of:
1. Existing SSTCN 102-class raw archives (`--include-sstcn-words` for 63 word classes `39..101`,
   or `--include-sstcn-all` for all 102 classes)
2. Ingested external sources via `--source-dir` (Hugging Face `VOYA_VSL`, `ViSignLanguage-Video`,
   Kaggle/QIPEDC videos, Zenodo `VSL400`)
3. User's personal webcam calibration samples (`--include-personal`)

Applies:
- Label Harmonizer (maps raw labels + region tags onto `vsl_lexicon.json` `entry_id` + `topic_map`)
- Body-Centric Normalization v2 (`PREPROCESSING_V2`, default) or legacy raw v1
- SHA-256 exact deduplication & strict group/signer-isolated train/val/test splitting
- Exports `sequences.npy`, `labels.npy`, `classes.json`, `topics.json`, `manifest.json`,
  and `harmonization_report.json` fully compatible with `train_sstcn.py` & `evaluate_sstcn.py`.
"""

import argparse
import json
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.models.vsl_102_dictionary import SIGN_DICT_102
from backend.training.fetch_sstcn_data import sha256
from backend.training.label_harmonizer import LabelHarmonizer
from backend.training.sstcn_data import (
    DATA_ROOT,
    PREPROCESSING,
    canonical_sequence,
    load_source,
    sequence_hash,
    write_json,
)
from backend.vision.skeleton_norm import PREPROCESSING_V2, normalize_sequence_ctj_v2


def _iter_sstcn_raw_records(raw_root: Path, include_all_102: bool = False):
    """Yield raw samples from backend/data/sstcn/raw."""
    if not raw_root.exists():
        return
    min_cls = 0 if include_all_102 else 39
    for npy_path in sorted(raw_root.rglob("*.npy")):
        parts = npy_path.parts
        archive = parts[len(raw_root.parts)] if len(parts) > len(raw_root.parts) else "sstcn"
        is_test = archive == "test_set"
        stem = npy_path.stem
        # Expected filename: sign_XXX_vid_YY_ZZ
        toks = stem.split("_")
        aug_id = 0
        group = f"sstcn/{npy_path.parent.name}/{stem}"
        if len(toks) >= 5 and toks[0] == "sign" and toks[2] == "vid":
            try:
                aug_id = int(toks[4])
                group = f"sstcn/{toks[1]}/{toks[3]}"
            except ValueError:
                pass
        try:
            raw_seq, label_idx = load_source(npy_path)
            if label_idx is None:
                continue
            label_idx = int(label_idx)
            if label_idx < min_cls or label_idx >= 102:
                continue
            yield {
                "seq": raw_seq,
                "raw_label": SIGN_DICT_102[label_idx],
                "source": f"sstcn/{archive}/{npy_path.name}",
                "source_name": "sstcn_102",
                "group": group,
                "augmentation": aug_id,
                "preferred_split": "test" if is_test else "pending",
                "video_id": None,
                "signer_id": None,
            }
        except Exception:
            continue


def _iter_ingested_source(source_dir: Path):
    """Yield samples from an ingested source directory containing `records.jsonl`."""
    rec_path = source_dir / "records.jsonl"
    if not rec_path.exists():
        raise FileNotFoundError(f"Missing records.jsonl in {source_dir}")
    with rec_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            npy_path = source_dir / rec["file"]
            if not npy_path.exists():
                continue
            yield {
                "seq": np.load(npy_path, allow_pickle=False),
                "raw_label": rec["raw_label"],
                "source": f"{rec.get('source_name', source_dir.name)}/{source_dir.name}/{rec['file']}",
                "source_name": rec.get("source_name", source_dir.name),
                "group": rec.get("group") or f"{source_dir.name}/{npy_path.stem}",
                "augmentation": 0,
                "preferred_split": "test" if rec.get("split") == "test" else "pending",
                "video_id": rec.get("video_id"),
                "signer_id": rec.get("signer_id"),
            }


def _iter_personal_calibration(cal_dir: Path):
    """Yield user-recorded webcam samples from `backend/data/personal_calibration/`."""
    if not cal_dir.exists():
        return
    for npy_path in sorted(cal_dir.rglob("*.npy")):
        meta_path = npy_path.with_suffix(".json")
        label = npy_path.parent.name
        signer = "personal"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                label = m.get("label") or label
                signer = m.get("signer_id") or signer
            except Exception:
                pass
        yield {
            "seq": np.load(npy_path, allow_pickle=False),
            "raw_label": label,
            "source": f"personal/{npy_path.parent.name}/{npy_path.name}",
            "source_name": "personal_calibration",
            "group": f"personal/{npy_path.parent.name}/{npy_path.stem}",
            "augmentation": 0,
            "preferred_split": "pending",
            "video_id": None,
            "signer_id": signer,
        }


def _assign_splits(by_class, kept_keys, rng, val_fraction, test_fraction):
    """Group recording aliases and known signers before assigning any class to a split."""
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for key in kept_keys:
        for row in by_class[key]:
            group = (row["source_name"], "clip", row["group"])
            row["component"] = group
            if row["signer_id"]:
                signer = (row["source_name"], "signer", str(row["signer_id"]))
                parent[root(group)] = root(signer)

    groups_by_class = defaultdict(set)
    classes_by_group = defaultdict(set)
    fixed_test = set()
    for key in kept_keys:
        for row in by_class[key]:
            group = row["component"] = root(row["component"])
            groups_by_class[key].add(group)
            classes_by_group[group].add(key)
            if row["split"] == "test":
                fixed_test.add(group)

    assignments = dict.fromkeys(fixed_test, "test")
    train_counts = {key: len(groups - fixed_test) for key, groups in groups_by_class.items()}
    for key in sorted(kept_keys, key=lambda key: (train_counts[key], key)):
        groups = groups_by_class[key]
        candidates = sorted(groups - fixed_test)
        rng.shuffle(candidates)
        targets = [("val", val_fraction)]
        if len(groups) >= 3 and not groups & fixed_test:
            targets.append(("test", test_fraction))
        for split, fraction in targets:
            if fraction <= 0:
                continue
            target = max(1, round(len(candidates) * fraction))
            assigned = sum(assignments.get(group) == split for group in groups)
            for group in candidates:
                if assigned >= target:
                    break
                affected = classes_by_group[group]
                if group in assignments or any(train_counts[k] <= 1 for k in affected):
                    continue
                assignments[group] = split
                assigned += 1
                for k in affected:
                    train_counts[k] -= 1

    return assignments


def prepare_unified(
    output_dir: Path,
    source_dirs: Optional[List[Path]] = None,
    include_sstcn_words: bool = False,
    include_sstcn_all: bool = False,
    sstcn_raw_root: Path = DATA_ROOT / "raw",
    include_personal: bool = False,
    personal_dir: Path = Path("backend/data/personal_calibration"),
    preprocessing_version: int = 2,
    min_samples_per_class: int = 2,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
    split_regions: bool = True,
    overwrite: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        if not (output_dir / "manifest.json").exists():
            raise ValueError(f"{output_dir} is not a generated dataset directory; refusing to overwrite")

    harmonizer = LabelHarmonizer(split_regions=split_regions)
    sources = []

    if include_sstcn_words or include_sstcn_all:
        sources.append(_iter_sstcn_raw_records(sstcn_raw_root, include_all_102=include_sstcn_all))
    for sdir in (source_dirs or []):
        sources.append(_iter_ingested_source(Path(sdir)))
    if include_personal:
        sources.append(_iter_personal_calibration(Path(personal_dir)))

    # Step 1: Canonicalize sequences, harmonize labels, deduplicate by SHA-256
    by_class_key: Dict[str, List[dict]] = defaultdict(list)
    class_meta: Dict[str, dict] = {}
    seen_hashes: Dict[str, str] = {}
    duplicates = []
    invalid_count = 0

    for item in chain.from_iterable(sources):
        try:
            seq_ctj = canonical_sequence(item["seq"])
            if preprocessing_version >= 2:
                seq_ctj = normalize_sequence_ctj_v2(seq_ctj)
            digest = sequence_hash(seq_ctj)
        except Exception:
            invalid_count += 1
            continue

        h_info = harmonizer.resolve(item["raw_label"], video_id=item.get("video_id"))
        ckey = h_info["class_key"]

        if digest in seen_hashes:
            duplicates.append({"source": item["source"], "sha256": digest, "first_class": seen_hashes[digest]})
            continue
        seen_hashes[digest] = ckey

        class_meta.setdefault(ckey, h_info)
        by_class_key[ckey].append({
            "seq": seq_ctj,
            "sha256": digest,
            "source": item["source"],
            "source_name": item["source_name"],
            "group": item["group"],
            "augmentation": int(item.get("augmentation", 0)),
            "split": item.get("preferred_split", "pending"),
            "signer_id": item.get("signer_id"),
            "class_key": ckey,
        })

    # Step 2: Filter classes meeting min_samples_per_class
    kept_keys = sorted([k for k, recs in by_class_key.items() if len(recs) >= min_samples_per_class])
    dropped_classes = [
        {"class_key": k, "label": class_meta[k]["canonical_label"], "count": len(by_class_key[k])}
        for k in sorted(by_class_key.keys())
        if k not in kept_keys
    ]
    if not kept_keys:
        raise ValueError(f"No classes have >= {min_samples_per_class} valid samples.")

    key_to_idx = {k: i for i, k in enumerate(kept_keys)}
    class_names = [class_meta[k]["canonical_label"] for k in kept_keys]
    entry_ids = [class_meta[k]["entry_id"] for k in kept_keys]
    topic_map: Dict[str, List[int]] = defaultdict(list)
    for k, idx in key_to_idx.items():
        topic_map[class_meta[k]["topic"]].append(idx)

    # Step 3: Grouped train / val / test split per class
    rng = np.random.default_rng(seed)
    assignments = _assign_splits(by_class_key, kept_keys, rng, val_fraction, test_fraction)
    final_records = []
    for k in kept_keys:
        for r in by_class_key[k]:
            r["split"] = assignments.get(r["component"], "train")
            # Exclude offline augmented siblings from val/test to prevent leakage
            if r["split"] != "train" and r["augmentation"] != 0:
                continue
            r["label"] = key_to_idx[k]
            final_records.append(r)

    # Step 4: Write mmap-friendly arrays and manifests
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.array([r["label"] for r in final_records], dtype=np.int64)

    seq_path = output_dir / "sequences.npy"
    lbl_path = output_dir / "labels.npy"
    shape = (len(final_records), 3, 30, 75)
    sequences = np.lib.format.open_memmap(seq_path, mode="w+", dtype=np.float32, shape=shape)
    for idx, record in enumerate(final_records):
        sequences[idx] = record.pop("seq")
    sequences.flush()
    del sequences
    np.save(lbl_path, labels, allow_pickle=False)

    manifest_samples = []
    counts = {"train": 0, "val": 0, "test": 0}
    sources_breakdown: Dict[str, int] = defaultdict(int)
    for row_idx, r in enumerate(final_records):
        counts[r["split"]] = counts.get(r["split"], 0) + 1
        sources_breakdown[r["source_name"]] += 1
        manifest_samples.append({
            "row": row_idx,
            "label": int(r["label"]),
            "split": r["split"],
            "group": r["group"],
            "signer_id": r["signer_id"],
            "split_group": r["component"],
            "augmentation": r["augmentation"],
            "source": r["source"],
            "source_name": r["source_name"],
            "sha256": r["sha256"],
        })

    preprocessing_cfg = PREPROCESSING_V2 if preprocessing_version >= 2 else PREPROCESSING
    classes_doc = {
        "classes": [
            {
                "index": i,
                "label": class_names[i],
                "entry_id": entry_ids[i],
                "region": class_meta[kept_keys[i]]["region"],
                "topic": class_meta[kept_keys[i]]["topic"],
            }
            for i in range(len(kept_keys))
        ],
        "word_class_indices": list(range(len(kept_keys))),
        "unknown": {"index": len(kept_keys), "is_model_class": False, "label": "UNKNOWN"},
    }
    write_json(output_dir / "classes.json", classes_doc)
    write_json(output_dir / "topics.json", dict(sorted(topic_map.items())))

    seq_sha = sha256(seq_path)
    lbl_sha = sha256(lbl_path)
    manifest = {
        "schema_version": 2,
        "generated_by": "prepare_unified_dataset.py",
        "seed": seed,
        "preprocessing": preprocessing_cfg,
        "num_classes": len(kept_keys),
        "class_names": class_names,
        "counts": counts,
        "sources_breakdown": dict(sources_breakdown),
        "duplicates_removed": len(duplicates),
        "invalid_removed": invalid_count,
        "sequences_sha256": seq_sha,
        "labels_sha256": lbl_sha,
        "files": {
            "sequences.npy": seq_sha,
            "labels.npy": lbl_sha,
        },
        "topic_map": dict(sorted(topic_map.items())),
        "samples": manifest_samples,
    }
    write_json(output_dir / "manifest.json", manifest)

    report = {
        "total_classes_kept": len(kept_keys),
        "matched_lexicon_classes": sum(1 for k in kept_keys if class_meta[k]["matched_lexicon"]),
        "unmatched_lexicon_classes": [
            class_meta[k]["canonical_label"] for k in kept_keys if not class_meta[k]["matched_lexicon"]
        ],
        "dropped_sparse_classes": dropped_classes,
        "topics_summary": {t: len(idxs) for t, idxs in sorted(topic_map.items())},
        "counts": counts,
        "sources_breakdown": dict(sources_breakdown),
    }
    write_json(output_dir / "harmonization_report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("backend/data/vsl_unified/prepared"))
    p.add_argument("--source-dir", type=Path, action="append", default=[], help="Thư mục nguồn đã ingest (có thể truyền nhiều lần)")
    p.add_argument("--include-sstcn-words", action="store_true", help="Gộp 63 lớp từ động (39..101) từ backend/data/sstcn/raw")
    p.add_argument("--include-sstcn-all", action="store_true", help="Gộp đủ 102 lớp từ backend/data/sstcn/raw")
    p.add_argument("--sstcn-raw-root", type=Path, default=DATA_ROOT / "raw")
    p.add_argument("--include-personal", action="store_true", help="Gộp mẫu tự quay từ backend/data/personal_calibration")
    p.add_argument("--personal-dir", type=Path, default=Path("backend/data/personal_calibration"))
    p.add_argument("--preprocessing-version", type=int, choices=[1, 2], default=2)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-split-regions", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    report = prepare_unified(
        output_dir=args.output,
        source_dirs=args.source_dir,
        include_sstcn_words=args.include_sstcn_words,
        include_sstcn_all=args.include_sstcn_all,
        sstcn_raw_root=args.sstcn_raw_root,
        include_personal=args.include_personal,
        personal_dir=args.personal_dir,
        preprocessing_version=args.preprocessing_version,
        min_samples_per_class=args.min_samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_regions=not args.no_split_regions,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
