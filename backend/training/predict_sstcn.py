"""Replay a skeleton or video through SSTCN without starting the web application."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.training.evaluate_sstcn import load_checkpoint_model
from backend.training.prepare_sstcn import VIDEO_EXTENSIONS, video_to_landmarks
from backend.training.sstcn_data import CLASS_NAMES, canonical_sequence, load_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[1] / "models/weights/sign_sstcn_attention_model.pth")
    p.add_argument("--device", default="cpu")
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.7)
    a = p.parse_args()
    if a.frame_stride < 1 or not 0 <= a.threshold <= 1:
        p.error("frame-stride must be positive and threshold must be in [0,1]")
    if a.input.suffix.lower() in VIDEO_EXTENSIONS:
        sequence, metadata = video_to_landmarks(a.input, a.frame_stride)
    else:
        sequence, label = load_source(a.input)
        metadata = {"embedded_label": int(label) if label is not None else None}
    sequence = canonical_sequence(sequence)
    model, _ = load_checkpoint_model(a.checkpoint, a.device)
    with torch.inference_mode():
        probabilities, indices = model(torch.from_numpy(sequence[None]).to(a.device)).softmax(1).topk(5)
    top5 = [{"index": int(i), "label": CLASS_NAMES[i], "probability": float(v)}
            for i, v in zip(indices[0].cpu().tolist(), probabilities[0].cpu().tolist())]
    print(json.dumps({"input": str(a.input), "metadata": metadata, "top5": top5,
                      "accepted_label": top5[0]["label"] if top5[0]["probability"] >= a.threshold else "UNKNOWN"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
