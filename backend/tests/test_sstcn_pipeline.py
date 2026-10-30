"""Regression checks for data leakage, legacy import, metrics and resumable training."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

import numpy as np
import torch

from backend.models.sstcn_model import load_sstcn_model
from backend.training.evaluate_sstcn import SequenceDataset, metrics_from_confusion
from backend.training.fetch_sstcn_data import fetch, safe_extract
from backend.training.prepare_sstcn import prepare, reconcile_groups
from backend.training.sstcn_data import canonical_sequence, grouped_split, load_prepared, load_source
from backend.training.train_sstcn import train


def sample(root, label, recording, augmentation=0, test=False, value=None):
    folder = root / ("test_set" if test else "data_01") / f"sign_{label:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sign_{label:03d}_vid_{recording:02d}_{augmentation:02d}.npy"
    if value is None:
        value = np.random.default_rng(label * 1000 + recording * 10 + augmentation).normal(0.2, 0.1, (30, 75, 3)).astype(np.float32)
    np.save(path, {"keypoints": value, "label": label})
    return path


class SSTCNDataTests(unittest.TestCase):
    def test_layout_and_resampling(self):
        value = np.arange(4 * 75 * 3, dtype=np.float32).reshape(4, 75, 3)
        result = canonical_sequence(value)
        self.assertEqual(result.shape, (3, 30, 75))
        np.testing.assert_array_equal(result[:, 0], value[0].T)
        np.testing.assert_array_equal(result[:, -1], value[-1].T)
        np.testing.assert_array_equal(result, canonical_sequence(value.transpose(2, 0, 1)))
        np.testing.assert_array_equal(result, canonical_sequence(value.reshape(4, 225)))

    def test_invalid_sequences_are_rejected(self):
        for value in (np.zeros((30, 75, 3)), np.full((30, 75, 3), np.nan), np.ones((30, 21, 3))):
            with self.assertRaises(ValueError):
                canonical_sequence(value)

    def test_legacy_dictionary_and_forbidden_pickle(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = sample(root, 5, 1)
            value, label = load_source(p)
            self.assertEqual(label, 5)
            self.assertEqual(value.shape, (30, 75, 3))
            class Payload:
                def __reduce__(self):
                    return eval, ("1+1",)
            np.save(root / "bad.npy", {"keypoints": Payload(), "label": 0})
            with self.assertRaises(pickle.UnpicklingError):
                load_source(root / "bad.npy")

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "bad.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("../escape.txt", "x")
            with self.assertRaises(ValueError):
                safe_extract(p, Path(t) / "extract")
            self.assertFalse((Path(t) / "escape.txt").exists())

    def test_grouped_split_is_reproducible(self):
        records = [{"label": label, "group": f"{label}/{video}", "split": "pending", "augmentation": aug}
                   for label in range(2) for video in range(6) for aug in range(4)]
        first = grouped_split([r.copy() for r in records], seed=7)
        second = grouped_split([r.copy() for r in records], seed=7)
        self.assertEqual(first, second)
        for group in {r["group"] for r in first}:
            self.assertEqual(len({r["split"] for r in first if r["group"] == group}), 1)

    def test_prepare_preserves_test_and_excludes_validation_augmentations(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            root = Path(t)
            for label in range(2):
                for recording in range(4):
                    for aug in range(2):
                        sample(root / "raw", label, recording, aug)
            original = np.load(sample(root / "raw", 0, 99), allow_pickle=True).item()["keypoints"]
            sample(root / "raw", 0, 99, augmentation=1)
            sample(root / "raw", 0, 90, test=True, value=original)
            manifest = prepare(root / "raw", root / "prepared", seed=7)
            self.assertEqual(manifest["counts"]["test"], 1)
            self.assertEqual(len(manifest["duplicates"]), 1)
            self.assertEqual(len(manifest["excluded_test_overlap_groups"]), 2)
            self.assertTrue(all("vid_99" not in r["source"] for r in manifest["samples"]))
            self.assertTrue(all(r["augmentation"] == 0 for r in manifest["samples"] if r["split"] != "train"))
            x, _, _ = load_prepared(root / "prepared")
            self.assertEqual(len(x), sum(manifest["counts"].values()))
            del x
            with (root / "prepared/sequences.npy").open("r+b") as f:
                f.seek(-1, 2)
                f.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(root / "prepared")

    def test_overwrite_cannot_delete_unrelated_folder(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            sample(root / "raw", 0, 0)
            (root / "unrelated").mkdir()
            (root / "unrelated/keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "not a generated"):
                prepare(root / "raw", root / "unrelated", overwrite=True)
            self.assertTrue((root / "unrelated/keep.txt").exists())

    def test_duplicate_recording_aliases_share_a_group(self):
        records = [{"label": 0, "group": group, "sha256": digest, "split": "pending"}
                   for group, digest in [("a", "same"), ("b", "same"), ("b", "augmented")]]
        self.assertEqual(reconcile_groups(records), set())
        self.assertEqual({r["group"] for r in records}, {"a"})

    def test_augmentation_preserves_missing_joints(self):
        x = np.full((1, 3, 30, 75), 0.5, dtype=np.float32)
        x[:, :, :, :33] = 0
        ds = SequenceDataset(x, np.array([0]), [0], augment=True)
        first, _ = ds[0]
        second, _ = ds[0]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(torch.count_nonzero(first[:, :, :33]).item(), 0)
        ds.epoch = 1
        changed, _ = ds[0]
        self.assertFalse(torch.equal(first, changed))

    def test_confusion_metrics(self):
        matrix = np.zeros((102, 102), dtype=np.int64)
        matrix[0, 0], matrix[0, 1], matrix[1, 1] = 2, 1, 1
        result = metrics_from_confusion(matrix)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["per_class"][0]["recall"], 2 / 3)
        self.assertEqual(result["per_class"][1]["precision"], 0.5)
        self.assertEqual(result["represented_classes"], 2)

    def test_missing_explicit_checkpoint_does_not_fall_back(self):
        with self.assertRaises(RuntimeError):
            load_sstcn_model("/nonexistent/sstcn.pth")


class SSTCNTrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_cpu_training(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            root = Path(t)
            for label in range(2):
                for recording in range(4):
                    sample(root / "raw", label, recording)
            prepare(root / "raw", root / "prepared", seed=7)
            def args(output, epochs, resume=None):
                return argparse.Namespace(data=root / "prepared", output=output, epochs=epochs,
                    batch_size=2, lr=0.001, weight_decay=0.0001, dropout=0.3, seed=7,
                    patience=8, threads=1, device="cpu", amp=False, augment=True,
                    init_from=None, resume=resume, allow_partial_classes=True)
            train(args(root / "full", 2))
            train(args(root / "resumed", 1))
            train(args(root / "resumed", 2, root / "resumed/last.pth"))
            full = torch.load(root / "full/last.pth", weights_only=True)
            resumed = torch.load(root / "resumed/last.pth", weights_only=True)
            for name in full["model_state_dict"]:
                self.assertTrue(torch.equal(full["model_state_dict"][name], resumed["model_state_dict"][name]), name)
            self.assertTrue((root / "resumed/validation/confusion_matrix.png").exists())
            self.assertTrue((root / "resumed/learning_curves.png").exists())
            model = load_sstcn_model(root / "resumed/best.pth")
            self.assertTrue(model.has_weights)


class SSTCNDownloadTests(unittest.TestCase):
    def test_github_failure_falls_back_to_drive(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/zip"}
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("sample.txt", "dataset")
        entry = {"id": "source-id", "name": "data.zip", "github_release_url": "https://github.com/test/data.zip"}
        failure = urllib.error.HTTPError(entry["github_release_url"], 404, "Not Found", {}, None)
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            with patch("backend.training.fetch_sstcn_data.urllib.request.urlopen", side_effect=[failure, Response(data.getvalue())]) as request:
                result, _ = fetch(entry, t)
            self.assertTrue(zipfile.is_zipfile(result))
            self.assertEqual(request.call_count, 2)
            self.assertTrue(request.call_args_list[1].args[0].full_url.startswith("https://drive.usercontent.google.com/"))

    def test_explicit_github_source_does_not_silently_fall_back(self):
        entry = {"id": "source-id", "name": "data.zip", "github_release_url": "https://github.com/test/data.zip"}
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            with patch("backend.training.fetch_sstcn_data.urllib.request.urlopen", side_effect=urllib.error.URLError("unavailable")) as request:
                with self.assertRaises(urllib.error.URLError):
                    fetch(entry, t, source="github")
            self.assertEqual(request.call_count, 1)

    def test_cached_archive_checksum_is_enforced(self):
        with tempfile.TemporaryDirectory() as t:
            with zipfile.ZipFile(Path(t) / "data.zip", "w") as z:
                z.writestr("sample.txt", "dataset")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                fetch({"id": "source-id", "name": "data.zip", "sha256": "0" * 64}, t)


if __name__ == "__main__":
    unittest.main()
