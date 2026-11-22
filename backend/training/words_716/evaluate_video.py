"""
Independent Video Evaluation Tool for WordRecognizer716 Pilot 40.

Evaluates single sign videos or batches against trained WordRecognizer716 checkpoints.
Features:
- Compatible MediaPipe Holistic 75-landmark extraction (30 frames, active trimming, temporal resampling).
- Subprocess bridging to .venv-sstcn if MediaPipe is not in current environment.
- Softmax over active classes with Top-1, Top-3, confidence, and margin calculation.
- Uncertainty detection based on confidence and margin thresholds.
- Full metric calculation (Accuracy, Top-3 Accuracy, Confusion Matrix, Precision, Recall, Macro F1).
- Clear separation between Training Reference vs Independent Holdout evaluation modes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CKPT = ROOT / "backend" / "models" / "words_716_checkpoints" / "pilot_40" / "best.pth"
DEFAULT_CLASSES = ROOT / "backend" / "data" / "words_716" / "pilot_40_classes.json"
SSTCN_PYTHON = ROOT / ".venv-sstcn" / "Scripts" / "python.exe"


class VideoEvaluator:
    """
    Evaluator for WordRecognizer716 on video or pre-extracted skeleton sequences.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CKPT,
        classes_path: Union[str, Path] = DEFAULT_CLASSES,
        confidence_threshold: float = 0.15,
        margin_threshold: float = 0.03,
        device: str = "cpu",
        body_centric: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.classes_path = Path(classes_path)
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.device = torch.device(device)
        self.body_centric = body_centric

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.classes_path.is_file():
            raise FileNotFoundError(f"Classes config not found: {self.classes_path}")

        # Load active classes
        cfg_data = json.loads(self.classes_path.read_text(encoding="utf-8"))
        if "classes" in cfg_data:
            self.class_list = cfg_data["classes"]
        elif isinstance(cfg_data, list):
            self.class_list = cfg_data
        else:
            raise ValueError("Unsupported classes format in classes file")

        self.active_indices = [c["class_index"] for c in self.class_list]
        self.active_count = len(self.active_indices)
        self.idx_to_meta = {c["class_index"]: c for c in self.class_list}
        self.idx_to_sub = {c: i for i, c in enumerate(self.active_indices)}

        # Load model
        from backend.training.words_716.model import WordRecognizer716

        self.model = WordRecognizer716(
            num_classes=716,
            active_indices=self.active_indices,
            dropout=0.0,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.ckpt_meta = {
            "epoch": ckpt.get("epoch"),
            "phase": ckpt.get("phase"),
            "saved_loss": ckpt.get("loss"),
            "schema_version": ckpt.get("schema_version"),
        }

    def extract_skeleton_from_video(self, video_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """
        Extract [3, 30, 75] skeleton from a video file using the compatible Holistic pipeline.
        If mediapipe is not installed in the current interpreter, delegates to .venv-sstcn.
        """
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check if mediapipe is available in current interpreter
        try:
            # pyrefly: ignore[missing-import]
            import mediapipe
            has_mp = True
        except ImportError:
            has_mp = False

        if has_mp:
            from backend.training.extract_words_716_skeleton import (
                DEFAULT_CONFIG,
                _create_holistic_detector,
                extract_raw_holistic_from_video,
                compute_active_segment,
                handle_missing_landmarks,
                resample_sequence_by_timestamps,
            )
            _, detector = _create_holistic_detector(DEFAULT_CONFIG)
            try:
                raw_seq, timestamps, _ = extract_raw_holistic_from_video(video_path, detector, frame_stride=DEFAULT_CONFIG["mediapipe"]["frame_stride"])
            finally:
                detector.close()

            s_idx, e_idx, trim_meta = compute_active_segment(raw_seq, timestamps, DEFAULT_CONFIG)
            act_seq = raw_seq[s_idx : e_idx + 1]
            act_ts = timestamps[s_idx : e_idx + 1]
            cleaned_seq, valid, missing_meta = handle_missing_landmarks(act_seq, act_ts, DEFAULT_CONFIG)
            tensor_3x30x75 = resample_sequence_by_timestamps(cleaned_seq, act_ts, target_frames=DEFAULT_CONFIG["target_frames"])
            meta = {**trim_meta, **missing_meta, "valid": valid}
            return tensor_3x30x75, meta
        else:
            # Delegate to extraction bridge via .venv-sstcn
            return self._extract_via_subprocess(video_path)

    def _extract_via_subprocess(self, video_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Run extraction bridge in .venv-sstcn interpreter.
        """
        if not SSTCN_PYTHON.is_file():
            raise RuntimeError(f"MediaPipe environment not found at: {SSTCN_PYTHON}")

        bridge_script = (
            "import sys, json, numpy as np\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, r'" + str(ROOT) + "')\n"
            "from backend.training.extract_words_716_skeleton import (\n"
            "  DEFAULT_CONFIG, _create_holistic_detector, extract_raw_holistic_from_video,\n"
            "  compute_active_segment, handle_missing_landmarks, resample_sequence_by_timestamps\n"
            ")\n"
            "video_path = Path(sys.argv[1])\n"
            "_, detector = _create_holistic_detector(DEFAULT_CONFIG)\n"
            "try:\n"
            "    raw_seq, timestamps, _ = extract_raw_holistic_from_video(video_path, detector, frame_stride=DEFAULT_CONFIG['mediapipe']['frame_stride'])\n"
            "finally:\n"
            "    detector.close()\n"
            "s_idx, e_idx, trim_meta = compute_active_segment(raw_seq, timestamps, DEFAULT_CONFIG)\n"
            "act_seq = raw_seq[s_idx : e_idx + 1]\n"
            "act_ts = timestamps[s_idx : e_idx + 1]\n"
            "cleaned_seq, valid, missing_meta = handle_missing_landmarks(act_seq, act_ts, DEFAULT_CONFIG)\n"
            "tensor_3x30x75 = resample_sequence_by_timestamps(cleaned_seq, act_ts, target_frames=DEFAULT_CONFIG['target_frames'])\n"
            "meta = {**trim_meta, **missing_meta, 'valid': valid}\n"
            "out_npy = video_path.with_suffix('.tmp_eval.npy')\n"
            "np.save(out_npy, tensor_3x30x75)\n"
            "print(json.dumps({'meta': meta, 'npy': str(out_npy)}, ensure_ascii=False))\n"
        )

        res = subprocess.run(
            [str(SSTCN_PYTHON), "-c", bridge_script, str(video_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [l.strip() for l in res.stdout.splitlines() if l.strip().startswith("{")]
        if not lines:
            raise RuntimeError(f"Subprocess extraction failed: {res.stderr}")
        data = json.loads(lines[-1])
        npy_path = Path(data["npy"])
        tensor_3x30x75 = np.load(npy_path)
        try:
            npy_path.unlink()
        except Exception:
            pass
        return tensor_3x30x75, data["meta"]

    def evaluate_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        true_class_index: Optional[int] = None,
        source_identifier: str = "custom_sequence",
    ) -> Dict[str, Any]:
        """
        Evaluate a single [3, 30, 75] skeleton sequence.
        """
        if self.body_centric:
            try:
                from backend.training.words_716.normalization import normalize_body_centric
            except ImportError:
                from normalization import normalize_body_centric
            if isinstance(sequence, torch.Tensor):
                seq_np = sequence.cpu().numpy()
            else:
                seq_np = np.array(sequence, copy=True)
            seq_np, _ = normalize_body_centric(seq_np)
            tensor = torch.from_numpy(seq_np).float()
        elif isinstance(sequence, np.ndarray):
            tensor = torch.from_numpy(sequence).float()
        else:
            tensor = sequence.float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [1, 3, 30, 75]
        assert tensor.shape == (1, 3, 30, 75), f"Expected shape [1, 3, 30, 75], got {tensor.shape}"

        tensor = tensor.to(self.device)

        with torch.no_grad():
            masked_logits = self.model(tensor, apply_class_mask=True)
            active_logits = masked_logits[:, self.active_indices]  # [1, 40]
            probs = F.softmax(active_logits, dim=-1)[0]  # [40]

        top3_vals, top3_sub = torch.topk(probs, k=min(3, self.active_count))
        top3_logits, _ = torch.topk(active_logits[0], k=min(3, self.active_count))

        top3_list = []
        for r in range(len(top3_vals)):
            sub_i = top3_sub[r].item()
            cls_idx = self.active_indices[sub_i]
            meta = self.idx_to_meta[cls_idx]
            top3_list.append({
                "rank": r + 1,
                "class_index": cls_idx,
                "word": meta["word"],
                "entry_id": meta.get("entry_id", f"class_{cls_idx}"),
                "region": meta.get("selected_region", "UNKNOWN"),
                "probability": round(top3_vals[r].item(), 4),
                "logit": round(top3_logits[r].item(), 4),
            })

        top1 = top3_list[0]
        confidence = top1["probability"]
        margin_prob = round(top3_vals[0].item() - top3_vals[1].item(), 4) if len(top3_vals) > 1 else 1.0
        margin_logit = round(top3_logits[0].item() - top3_logits[1].item(), 4) if len(top3_logits) > 1 else 0.0

        # Uncertainty flags
        uncertainty_flags = []
        if confidence < self.confidence_threshold:
            uncertainty_flags.append(f"LOW_CONFIDENCE (< {self.confidence_threshold})")
        if margin_prob < self.margin_threshold:
            uncertainty_flags.append(f"LOW_MARGIN (< {self.margin_threshold})")
        is_uncertain = len(uncertainty_flags) > 0

        result = {
            "source_identifier": str(source_identifier),
            "top1_predicted_class_index": top1["class_index"],
            "top1_predicted_word": top1["word"],
            "confidence": round(confidence, 4),
            "margin_probability": margin_prob,
            "margin_logit": margin_logit,
            "is_uncertain": is_uncertain,
            "uncertainty_flags": uncertainty_flags,
            "top3": top3_list,
        }

        if true_class_index is not None:
            is_correct = (top1["class_index"] == true_class_index)
            top3_classes = [t["class_index"] for t in top3_list]
            top3_correct = (true_class_index in top3_classes)
            true_rank = top3_classes.index(true_class_index) + 1 if top3_correct else -1
            true_meta = self.idx_to_meta.get(true_class_index, {})

            result["ground_truth"] = {
                "true_class_index": true_class_index,
                "true_word": true_meta.get("word", "UNKNOWN"),
                "is_correct": is_correct,
                "top3_correct": top3_correct,
                "rank_of_true_class": true_rank,
            }

        return result

    def evaluate_video(
        self,
        video_path: Union[str, Path],
        true_class_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full end-to-end evaluation: Video -> Skeleton -> Model -> Output Report.
        """
        video_path = Path(video_path)
        tensor_3x30x75, extraction_meta = self.extract_skeleton_from_video(video_path)
        res = self.evaluate_sequence(
            sequence=tensor_3x30x75,
            true_class_index=true_class_index,
            source_identifier=video_path.name,
        )
        res["extraction_metadata"] = extraction_meta
        return res

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        evaluation_mode: str = "independent_holdout",
    ) -> Dict[str, Any]:
        """
        Evaluate a list of sample dictionaries.
        Each sample dict must contain:
          - 'sequence': np.ndarray [3, 30, 75] OR 'video_path': str/Path
          - 'true_class_index': int
          - optional: 'signer_id', 'word', 'repetition'
        """
        if evaluation_mode not in ("training_reference", "independent_holdout"):
            raise ValueError("evaluation_mode must be 'training_reference' or 'independent_holdout'")

        sample_evals = []
        correct_count = 0
        top3_correct_count = 0
        uncertain_count = 0

        # Confusion tracking
        matrix_dim = self.active_count
        conf_matrix = [[0] * matrix_dim for _ in range(matrix_dim)]
        class_stats = {
            c: {"tp": 0, "fp": 0, "fn": 0, "word": self.idx_to_meta[c]["word"]}
            for c in self.active_indices
        }

        for idx, s in enumerate(samples):
            true_idx = int(s["true_class_index"])
            if "sequence" in s:
                seq = s["sequence"]
                ev = self.evaluate_sequence(
                    sequence=seq,
                    true_class_index=true_idx,
                    source_identifier=s.get("source_identifier", f"sample_{idx}"),
                )
            elif "video_path" in s:
                v_path = s["video_path"]
                ev = self.evaluate_video(
                    video_path=v_path,
                    true_class_index=true_idx,
                )
            else:
                raise ValueError(f"Sample #{idx} has neither 'sequence' nor 'video_path'")

            pred_idx = ev["top1_predicted_class_index"]
            is_corr = ev["ground_truth"]["is_correct"]
            is_top3_corr = ev["ground_truth"]["top3_correct"]

            if is_corr:
                correct_count += 1
            if is_top3_corr:
                top3_correct_count += 1
            if ev["is_uncertain"]:
                uncertain_count += 1

            if true_idx in self.idx_to_sub and pred_idx in self.idx_to_sub:
                r_true = self.idx_to_sub[true_idx]
                c_pred = self.idx_to_sub[pred_idx]
                conf_matrix[r_true][c_pred] += 1

                if is_corr:
                    class_stats[true_idx]["tp"] += 1
                else:
                    class_stats[true_idx]["fn"] += 1
                    class_stats[pred_idx]["fp"] += 1

            # Extra metadata preservation
            for extra_key in ("signer_id", "repetition", "condition"):
                if extra_key in s:
                    ev[extra_key] = s[extra_key]

            sample_evals.append(ev)

        n_samples = len(samples)
        acc = correct_count / n_samples if n_samples > 0 else 0.0
        top3_acc = top3_correct_count / n_samples if n_samples > 0 else 0.0
        uncertain_rate = uncertain_count / n_samples if n_samples > 0 else 0.0

        # Precision, Recall, F1
        f1_list = []
        for c, st in class_stats.items():
            tp, fp, fn = st["tp"], st["fp"], st["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            st["precision"] = round(prec, 4)
            st["recall"] = round(rec, 4)
            st["f1_score"] = round(f1, 4)
            if (tp + fn) > 0:  # Only average classes present in true labels
                f1_list.append(f1)

        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

        # Certain vs uncertain accuracy
        certain_evals = [e for e in sample_evals if not e["is_uncertain"]]
        certain_acc = (
            sum(1 for e in certain_evals if e["ground_truth"]["is_correct"]) / len(certain_evals)
            if certain_evals
            else None
        )

        batch_report = {
            "schema_version": "3.2d-video-evaluation-report-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_mode": evaluation_mode,
            "evaluation_context_note": (
                "TRAINING_REFERENCE: Evaluates training sample fitting/memorization capacity."
                if evaluation_mode == "training_reference"
                else "INDEPENDENT_HOLDOUT: Unseen cross-signer generalization test. Thresholds must not be tuned on this data."
            ),
            "checkpoint_evaluated": self.checkpoint_path.name,
            "checkpoint_info": self.ckpt_meta,
            "thresholds": {
                "confidence_threshold": self.confidence_threshold,
                "margin_threshold": self.margin_threshold,
            },
            "summary_metrics": {
                "total_samples": n_samples,
                "accuracy": round(acc, 4),
                "top3_accuracy": round(top3_acc, 4),
                "macro_f1_score": round(macro_f1, 4),
                "uncertain_samples_count": uncertain_count,
                "uncertainty_rate": round(uncertain_rate, 4),
                "accuracy_on_certain_subset": round(certain_acc, 4) if certain_acc is not None else None,
            },
            "class_metrics": class_stats,
            "samples": sample_evals,
        }
        return batch_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=None, help="Path to single sign video")
    parser.add_argument("--sequence", type=str, default=None, help="Path to pre-extracted .npy sequence [3, 30, 75]")
    parser.add_argument("--true-class", type=int, default=None, help="Ground truth class_index for single evaluation")
    parser.add_argument("--manifest", type=str, default=None, help="JSON/JSONL manifest for batch evaluation")
    parser.add_argument("--mode", choices=["training_reference", "independent_holdout"], default="independent_holdout")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--active-classes", type=str, default=str(DEFAULT_CLASSES))
    parser.add_argument("--conf-thresh", type=float, default=0.15)
    parser.add_argument("--margin-thresh", type=float, default=0.03)
    parser.add_argument("--body-centric", action="store_true", help="Apply Body-Centric Normalization to input sequence")
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    evaluator = VideoEvaluator(
        checkpoint_path=args.checkpoint,
        classes_path=args.active_classes,
        confidence_threshold=args.conf_thresh,
        margin_threshold=args.margin_thresh,
        body_centric=args.body_centric,
    )

    if args.video:
        res = evaluator.evaluate_video(args.video, true_class_index=args.true_class)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.sequence:
        arr = np.load(args.sequence)
        res = evaluator.evaluate_sequence(arr, true_class_index=args.true_class, source_identifier=args.sequence)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.manifest:
        m_path = Path(args.manifest)
        if m_path.suffix == ".jsonl":
            samples = [json.loads(l) for l in m_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            samples = json.loads(m_path.read_text(encoding="utf-8"))
        res = evaluator.evaluate_batch(samples, evaluation_mode=args.mode)
        print(f"Batch evaluation finished ({args.mode}):")
        print(f"  Accuracy: {res['summary_metrics']['accuracy']*100:.2f}%")
        print(f"  Top-3 Accuracy: {res['summary_metrics']['top3_accuracy']*100:.2f}%")
        print(f"  Macro F1: {res['summary_metrics']['macro_f1_score']:.4f}")
        print(f"  Uncertainty Rate: {res['summary_metrics']['uncertainty_rate']*100:.1f}%")
        if args.output:
            Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved full batch report to: {args.output}")
    else:
        print("Please provide --video, --sequence, or --manifest.")
        sys.exit(1)


if __name__ == "__main__":
    main()
