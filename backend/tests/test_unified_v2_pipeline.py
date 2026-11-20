"""End-to-end tests for Unified Multi-Source VSL Pipeline (Phases 0-4):
- Body-centric normalization v2 invariance (translation, scale, lower-body zeroing)
- KVSL 8-transform online augmentation safety
- LabelHarmonizer regional separation & lexicon matching
- VOYA_VSL multi-layout ingestion
- prepare_unified -> dynamic train_sstcn (v2) -> load_sstcn_model -> recognizer topic masking
- Personal calibration fine-tuning (including dynamic class expansion)
- FastAPI new endpoints (/api/sstcn/topics, /api/calibration/*, /api/recognize_sequence)
"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backend.models.sstcn_model import load_sstcn_model
from backend.training.augmentations import apply_kvsl_augmentations
from backend.training.colab_train_sstcn import create_gpu_bundle
from backend.training.finetune_personal import finetune_personal
from backend.training.ingest_voya_vsl import extract_75_joints_from_voya_array, ingest_from_directory
from backend.training.label_harmonizer import LabelHarmonizer
from backend.training.prepare_unified_dataset import prepare_unified
from backend.training.train_sstcn import train
from backend.vision.skeleton_norm import (
    normalize_sequence_ctj_v2,
    normalize_sequence_tjc_v2,
    resample_sequence_by_timestamps,
)


def _make_realistic_skeleton(seed: int, frames: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.2, 0.8, size=(frames, 75, 3)).astype(np.float32)
    # Set stable left/right shoulders (pose 11 and 12)
    seq[:, 11, :] = np.array([0.35, 0.40, 0.0], dtype=np.float32)
    seq[:, 12, :] = np.array([0.65, 0.40, 0.0], dtype=np.float32)
    return seq


class SkeletonNormAndAugmentTests(unittest.TestCase):
    def test_v2_translation_and_scale_invariance(self):
        base = _make_realistic_skeleton(1, frames=30)
        # Mark some hand joints missing (0)
        base[:, 33:40, :] = 0.0
        norm_base = normalize_sequence_tjc_v2(base)

        # Lower body (pose 23..32) must be strictly 0
        self.assertEqual(np.count_nonzero(norm_base[:, 23:33, :]), 0)
        # Missing hand joints must remain strictly 0
        self.assertEqual(np.count_nonzero(norm_base[:, 33:40, :]), 0)

        # Translate & uniformly scale all present joints
        transformed = base.copy()
        valid = np.any(transformed != 0.0, axis=-1)
        transformed[valid] = transformed[valid] * 2.3 + np.array([0.75, -0.42, 0.15], dtype=np.float32)
        norm_transformed = normalize_sequence_tjc_v2(transformed)

        np.testing.assert_allclose(norm_base, norm_transformed, atol=1e-5)

    def test_timestamp_resampling_handles_variable_fps(self):
        frames = np.linspace(0.0, 1.0, 18, dtype=np.float32)[:, None, None] * np.ones((1, 75, 3), dtype=np.float32)
        ts = np.linspace(1000.0, 3000.0, 18)
        resampled = resample_sequence_by_timestamps(frames, ts, target_len=30)
        self.assertEqual(resampled.shape, (30, 75, 3))
        np.testing.assert_allclose(resampled[0], frames[0], atol=1e-5)
        np.testing.assert_allclose(resampled[-1], frames[-1], atol=1e-5)

    def test_kvsl_augmentations_preserve_shape_and_missing_joints(self):
        seq = normalize_sequence_ctj_v2(_make_realistic_skeleton(2).transpose(2, 0, 1))
        seq[:, :, 33:45] = 0.0
        rng = np.random.default_rng(123)
        aug = apply_kvsl_augmentations(seq, rng, preprocessing_version=2)
        self.assertEqual(aug.shape, (3, 30, 75))
        self.assertEqual(aug.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(aug)))
        self.assertEqual(np.count_nonzero(aug[:, :, 33:45]), 0)


class LabelHarmonizerAndIngestTests(unittest.TestCase):
    def test_regional_separation_and_synonyms(self):
        h = LabelHarmonizer(split_regions=True)
        north = h.resolve("địa chỉ (bắc)")
        south = h.resolve("địa chỉ (nam)")
        self.assertNotEqual(north["class_key"], south["class_key"])
        self.assertEqual(north["region"], "B")
        self.assertEqual(south["region"], "N")

        greet = h.resolve("xin chao")
        self.assertTrue(greet["matched_lexicon"])
        self.assertTrue(bool(greet["entry_id"]))

    def test_voya_multi_layout_extraction_and_ingest(self):
        for dim in (1605, 1662, 1629, 225, 195):
            arr = np.ones((45, dim), dtype=np.float32) * 0.5
            out = extract_75_joints_from_voya_array(arr)
            self.assertEqual(out.shape, (45, 75, 3))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "voya_raw"
            (raw_dir / "Chào").mkdir(parents=True)
            np.savez_compressed(raw_dir / "Chào" / "s1.npz", sequence=np.ones((60, 1605), dtype=np.float32) * 0.4)
            out_dir = root / "voya_out"
            summary = ingest_from_directory(raw_dir, out_dir)
            self.assertEqual(summary["ingested"], 1)
            self.assertTrue((out_dir / "records.jsonl").exists())

    def test_voya_per_class_shard_loads_all_rows_and_maps_filename_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "data"
            (source / "Merged").mkdir(parents=True)
            (source / "labels.json").write_text('{"class_0001":"địa chỉ (bắc)"}', encoding="utf-8")
            np.savez_compressed(source / "Merged/class_0001.npz",
                sequences=np.ones((3, 60, 1605), dtype=np.float32), labels=np.ones(3, dtype=np.int32))
            output = root / "out"
            self.assertEqual(ingest_from_directory(source, output)["ingested"], 3)
            rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
            self.assertEqual({row["raw_label"] for row in rows}, {"địa chỉ (bắc)"})
            self.assertEqual(len({row["group"] for row in rows}), 3)


class UnifiedEndToEndV2Tests(unittest.TestCase):
    def test_signer_and_shared_recordings_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for i in range(12):
                file = f"clip_{i}.npy"
                np.save(root / file, _make_realistic_skeleton(i))
                rows.append({"file": file, "raw_label": ["Chào", "Cảm ơn"][i % 2],
                    "source_name": "vsl400", "group": f"clip_{i // 2}", "signer_id": f"s{i // 4}"})
            (root / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            prepare_unified(root / "out", source_dirs=[root])
            samples = json.loads((root / "out/manifest.json").read_text())["samples"]
            self.assertEqual({row["split"] for row in samples}, {"train", "val", "test"})
            for signer in {row["signer_id"] for row in samples}:
                self.assertEqual(len({row["split"] for row in samples if row["signer_id"] == signer}), 1)

    def test_prepare_train_load_topic_mask_and_personal_finetune(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            src_dir = root / "src_a"
            samples_dir = src_dir / "samples"
            samples_dir.mkdir(parents=True)

            raw_labels = ["Chào", "Cảm ơn", "Bác sĩ", "Gia đình", "địa chỉ (bắc)"]
            recs = []
            idx = 0
            for lbl in raw_labels:
                for g in range(4):
                    seq = _make_realistic_skeleton(100 + idx, frames=24)
                    fname = f"s_{idx:03d}.npy"
                    np.save(samples_dir / fname, seq)
                    recs.append({
                        "file": f"samples/{fname}",
                        "raw_label": lbl,
                        "source_name": "synthetic_voya",
                        "group": f"syn/{lbl}/g{g}",
                    })
                    idx += 1
            (src_dir / "records.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                encoding="utf-8",
            )

            prepared_dir = root / "prepared_v2"
            report = prepare_unified(
                output_dir=prepared_dir,
                source_dirs=[src_dir],
                preprocessing_version=2,
                min_samples_per_class=2,
                seed=7,
            )
            self.assertEqual(report["total_classes_kept"], 5)
            self.assertTrue((prepared_dir / "topics.json").exists())

            # Package GPU bundle check
            bundle = create_gpu_bundle(prepared_dir, root / "gpu_bundle.zip")
            self.assertTrue(bundle.exists())

            # Train dynamic 5-class v2 model with KVSL augmentation + class balance + cosine LR
            run_dir = root / "run_v2"
            train_args = argparse.Namespace(
                data=prepared_dir,
                output=run_dir,
                epochs=2,
                batch_size=4,
                lr=1e-3,
                weight_decay=1e-4,
                dropout=0.2,
                seed=7,
                patience=5,
                threads=1,
                device="cpu",
                amp=False,
                augment=True,
                augment_profile="kvsl",
                class_balanced=True,
                label_smoothing=0.1,
                scheduler="cosine",
                init_from=None,
                resume=None,
                allow_partial_classes=True,
            )
            train(train_args)

            best_ckpt = run_dir / "best.pth"
            model = load_sstcn_model(weights_path=str(best_ckpt), device="cpu")
            self.assertEqual(model.num_classes, 5)
            self.assertEqual(model.preprocessing_version, 2)
            self.assertEqual(len(model.class_names), 5)
            self.assertTrue(len(model.topic_map) >= 1)

            # Test Personal Calibration fine-tune adding a 6th brand-new word
            cal_dir = root / "personal_cal" / "tu_moi_cua_toi"
            cal_dir.mkdir(parents=True)
            for k in range(3):
                np.save(cal_dir / f"sample_{k}.npy", _make_realistic_skeleton(500 + k, frames=30))
                (cal_dir / f"sample_{k}.json").write_text(
                    json.dumps({"label": "Từ mới của tôi"}, ensure_ascii=False), encoding="utf-8"
                )

            ft_out = root / "sign_sstcn_personal.pth"
            ft_res = finetune_personal(
                base_checkpoint=best_ckpt,
                personal_dir=root / "personal_cal",
                output_path=ft_out,
                replay_data=prepared_dir,
                replay_per_class=2,
                epochs=2,
                repeat_factor=2,
                seed=7,
            )
            self.assertEqual(ft_res["num_classes"], 6)
            self.assertIn("Từ mới của tôi", ft_res["added_classes"])

            ft_model = load_sstcn_model(weights_path=str(ft_out), device="cpu")
            self.assertEqual(ft_model.num_classes, 6)
            self.assertEqual(ft_model.preprocessing_version, 2)


if __name__ == "__main__":
    unittest.main()
