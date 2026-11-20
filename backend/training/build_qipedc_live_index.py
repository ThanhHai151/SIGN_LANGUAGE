"""Build a compact compressed index of QIPEDC live-recognition templates.

Usage:
    python -m backend.training.build_qipedc_live_index --scope common
    python -m backend.training.build_qipedc_live_index --scope all
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
import time

import numpy as np

from backend.models import video_cache
from backend.models.lexicon import load_lexicon, motion_clip
from backend.vision.qipedc_live import DEFAULT_WORDS, FRAMES, _variants
from backend.vision.qipedc_practice import TIME_SIGN_ENTRY, _visible_clip, reference_path

COARSE_BINS = 6
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference" / "live_index.npz"
SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"


def _video_id_of(entry):
    m = video_cache.VIDEO_URL_RE.search(entry.get("video_url") or "")
    return m.group(1) if m else None


def select_entries(scope="common"):
    lex = load_lexicon()
    by_id = lex["by_id"]
    selected = []
    seen = set()

    def add(entry):
        if not entry:
            return
        eid = entry["id"]
        if eid in seen or eid == TIME_SIGN_ENTRY:
            return
        vid = _video_id_of(entry)
        if not vid:
            return
        seen.add(eid)
        selected.append((eid, vid, entry["word"]))

    for eid in DEFAULT_WORDS:
        add(by_id.get(eid))

    for entry in lex["entries"]:
        if entry.get("fingerspelling"):
            continue
        if scope == "common" and not entry.get("common"):
            continue
        add(entry)
    return selected


def _existing_npy(video_id):
    seed = SEED_DIR / f"v3_{video_id}.npy"
    if seed.is_file():
        return seed
    cached = Path(video_cache.CACHE_DIR) / "practice" / f"v3_{video_id}.npy"
    if cached.is_file():
        return cached
    return None


def _download_one(video_id):
    if _existing_npy(video_id) is not None:
        return video_id, True
    path = video_cache.ensure_cached("videos", video_id, timeout=25)
    return video_id, bool(path)


def _extract_one(video_id):
    existing = _existing_npy(video_id)
    if existing is not None:
        return video_id, str(existing), None
    video = video_cache.cache_path("videos", video_id)
    if not os.path.isfile(video):
        return video_id, None, "video_not_cached"
    try:
        from backend.training.ingest_videos import extract_video_to_75_joints

        active = motion_clip(video_id)
        try:
            frames = extract_video_to_75_joints(Path(video), target_len=None, active_window=active, webcam_profile=True)
            _visible_clip(frames)
        except ValueError:
            if not active:
                raise
            frames = extract_video_to_75_joints(Path(video), target_len=None, webcam_profile=True)
            _visible_clip(frames)
        out = Path(video_cache.CACHE_DIR) / "practice" / f"v3_{video_id}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.npy")
        np.save(tmp, frames.astype(np.float32), allow_pickle=False)
        tmp.replace(out)
        return video_id, str(out), None
    except Exception as exc:  # noqa: BLE001
        return video_id, None, str(exc)


def _coarse_pool(features, observed):
    """Pool [V, 24, H, 14] into [V, 6, H, 14] masked temporal means for fast Stage-1 filtering."""
    v, t, h, d = features.shape
    step = t // COARSE_BINS
    f = features.reshape(v, COARSE_BINS, step, h, d)
    o = observed.reshape(v, COARSE_BINS, step, h, 1).astype(np.float32)
    denom = np.maximum(1.0, o.sum(axis=2))
    return ((f * o).sum(axis=2) / denom).astype(np.float16)


def pack_index(items, output_path: Path):
    buckets = {
        1: {"ids": [], "features": [], "observed": [], "penalties": [], "var_to_entry": []},
        2: {"ids": [], "features": [], "observed": [], "penalties": [], "var_to_entry": []},
    }
    skipped = 0
    for entry_id, npy_path in items:
        try:
            arr = np.load(npy_path, allow_pickle=False)
            variants = _variants(arr, 1.0, reference=True)
        except Exception:  # noqa: BLE001
            variants = None
        if variants is None:
            skipped += 1
            continue
        features, observed, penalties = variants
        hands = features.shape[2]
        if hands not in buckets:
            skipped += 1
            continue
        bucket = buckets[hands]
        idx = len(bucket["ids"])
        bucket["ids"].append(entry_id)
        bucket["features"].append(features.astype(np.float16))
        bucket["observed"].append(observed.astype(bool))
        bucket["penalties"].append(penalties.astype(np.float16))
        bucket["var_to_entry"].append(np.full(len(features), idx, dtype=np.int32))

    arrays = {}
    for hands, prefix in ((1, "uni"), (2, "bi")):
        b = buckets[hands]
        if b["ids"]:
            feats = np.concatenate(b["features"], axis=0)
            obs = np.concatenate(b["observed"], axis=0)
            pens = np.concatenate(b["penalties"], axis=0)
            v2e = np.concatenate(b["var_to_entry"], axis=0)
            coarse = _coarse_pool(feats.astype(np.float32), obs)
        else:
            feats = np.zeros((0, FRAMES, hands, 14), dtype=np.float16)
            obs = np.zeros((0, FRAMES, hands), dtype=bool)
            pens = np.zeros((0,), dtype=np.float16)
            v2e = np.zeros((0,), dtype=np.int32)
            coarse = np.zeros((0, COARSE_BINS, hands, 14), dtype=np.float16)
        arrays[f"{prefix}_ids"] = np.array(b["ids"], dtype=str)
        arrays[f"{prefix}_features"] = feats
        arrays[f"{prefix}_observed"] = obs
        arrays[f"{prefix}_penalties"] = pens
        arrays[f"{prefix}_var_to_entry"] = v2e
        arrays[f"{prefix}_coarse"] = coarse

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".part.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(output_path)
    return len(buckets[1]["ids"]), len(buckets[2]["ids"]), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["common", "all"], default="common")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--extract-workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    entries = select_entries(args.scope)
    if args.limit > 0:
        entries = entries[:args.limit]
    unique_vids = list(dict.fromkeys(vid for _, vid, _ in entries))
    print(f"[index] Selected {len(entries)} entries ({len(unique_vids)} unique videos, scope={args.scope})", flush=True)

    # 1. Parallel download missing QIPEDC videos
    missing_vids = [v for v in unique_vids if _existing_npy(v) is None and not os.path.isfile(video_cache.cache_path("videos", v))]
    print(f"[index] Downloading {len(missing_vids)} missing videos...", flush=True)
    dl_ok = 0
    if missing_vids:
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            for i, (vid, ok) in enumerate(pool.map(_download_one, missing_vids), start=1):
                if ok:
                    dl_ok += 1
                if i % 50 == 0 or i == len(missing_vids):
                    print(f"[index] Downloaded {i}/{len(missing_vids)} (ok={dl_ok}, elapsed={time.perf_counter()-t0:.1f}s)", flush=True)

    # 2. Parallel extract missing .npy keypoints
    to_extract = [v for v in unique_vids if _existing_npy(v) is None]
    print(f"[index] Extracting keypoints for {len(to_extract)} videos (workers={args.extract_workers})...", flush=True)
    vid_to_npy = {v: str(p) for v in unique_vids if (p := _existing_npy(v)) is not None}
    ext_err = 0
    if to_extract:
        with ProcessPoolExecutor(max_workers=args.extract_workers) as pool:
            for i, (vid, npy_path, err) in enumerate(pool.map(_extract_one, to_extract), start=1):
                if npy_path:
                    vid_to_npy[vid] = npy_path
                else:
                    ext_err += 1
                if i % 50 == 0 or i == len(to_extract):
                    print(f"[index] Extracted {i}/{len(to_extract)} (total_npy={len(vid_to_npy)}, err={ext_err}, elapsed={time.perf_counter()-t0:.1f}s)", flush=True)

    # 3. Pack compressed index
    items = [(eid, vid_to_npy[vid]) for eid, vid, _ in entries if vid in vid_to_npy]
    n_uni, n_bi, skipped = pack_index(items, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(
        f"[index] Done in {time.perf_counter()-t0:.1f}s -> {args.output} "
        f"({size_mb:.2f} MB, uni={n_uni}, bi={n_bi}, total={n_uni+n_bi}, skipped={skipped})",
        flush=True,
    )


if __name__ == "__main__":
    main()
