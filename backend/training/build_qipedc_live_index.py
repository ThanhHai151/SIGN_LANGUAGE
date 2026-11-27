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
from backend.vision.qipedc_live import DEFAULT_WORDS, FEATURE_DIM, FRAMES, _variants
from backend.vision.qipedc_practice import _visible_clip

COARSE_BINS = 6
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference" / "live_index.npz"
SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
_WORKER_HOLISTIC = None


def _init_extract_worker():
    global _WORKER_HOLISTIC
    from backend.training.ingest_videos import _import_mediapipe_holistic

    mp_holistic = _import_mediapipe_holistic()
    _WORKER_HOLISTIC = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.30,
    )


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
        if eid in seen:
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
        if entry.get("sstcn_index") is not None and int(entry["sstcn_index"]) < 39:
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
            frames = extract_video_to_75_joints(
                Path(video), target_len=None, active_window=active, webcam_profile=True, holistic_instance=_WORKER_HOLISTIC
            )
            _visible_clip(frames)
        except ValueError:
            if not active:
                raise
            frames = extract_video_to_75_joints(
                Path(video), target_len=None, webcam_profile=True, holistic_instance=_WORKER_HOLISTIC
            )
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
    """Pool [V, 24, H, D] into [V, 6, H, D] masked temporal means for fast Stage-1 filtering."""
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
            feats = np.zeros((0, FRAMES, hands, FEATURE_DIM), dtype=np.float16)
            obs = np.zeros((0, FRAMES, hands), dtype=bool)
            pens = np.zeros((0,), dtype=np.float16)
            v2e = np.zeros((0,), dtype=np.int32)
            coarse = np.zeros((0, COARSE_BINS, hands, FEATURE_DIM), dtype=np.float16)
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


def sync_qipedc_dictionary():
    """Fetch latest metadata from https://qipedc.moet.gov.vn/dictionary/getAll and update local dictionary if changed."""
    import json
    from tempfile import TemporaryDirectory
    from unicodedata import normalize
    import urllib.request

    from backend.training.build_lexicon import OUT_PATH, SRC_DIR, build

    def clean_row(row):
        video_id, word, part_of_speech, definition, entry_type, has_image = row
        return [
            video_id, normalize("NFC", word or "").strip(), normalize("NFC", part_of_speech or "").strip(),
            normalize("NFC", definition or "").strip(), entry_type, bool(has_image),
        ]

    src_path = Path(SRC_DIR) / "qipedc_dictionary.json"
    req = urllib.request.Request(
        "https://qipedc.moet.gov.vn/dictionary/getAll",
        data=b"group=100000&text=",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    online_by_id = {
        row["_id"]: clean_row((
            row["_id"], row.get("word"), row.get("tl"), row.get("description"), row.get("type", 0), row.get("i"),
        ))
        for row in payload.get("data") or () if row.get("_id")
    }
    if not online_by_id:
        return 0, False
    local_rows = json.loads(src_path.read_text(encoding="utf-8")) if src_path.is_file() else []
    local_by_id = {row[0]: clean_row(row) for row in local_rows}
    if online_by_id == local_by_id:
        return len(online_by_id), False

    # Keep existing ordering because duplicate-word IDs and their ranking depend on it.
    row_order = dict.fromkeys((*local_by_id, *online_by_id))
    updated = [online_by_id[video_id] for video_id in row_order if video_id in online_by_id]
    lexicon_path = Path(OUT_PATH)
    with TemporaryDirectory(prefix="qipedc-sync-", dir=lexicon_path.parent) as staging:
        staging = Path(staging)
        staged_source = staging / src_path.name
        staged_lexicon = staging / lexicon_path.name
        staged_source.write_text(json.dumps(updated, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        build(dictionary_path=staged_source, output_path=staged_lexicon)
        staged_lexicon.replace(lexicon_path)
        staged_source.replace(src_path)
    load_lexicon.cache_clear()
    return len(updated), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["common", "all"], default="common")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--extract-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--extract-only", action="store_true", help="Only download & extract .npy files without packing index")
    parser.add_argument("--sync-dictionary", action="store_true", help="Sync dictionary metadata from qipedc.moet.gov.vn first")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.sync_dictionary:
        try:
            total_rows, updated = sync_qipedc_dictionary()
            print(f"[index] Synced QIPEDC dictionary: {total_rows} rows (updated={updated})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[index] Dictionary sync warning: {exc}", flush=True)

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
                if i % 100 == 0 or i == len(missing_vids):
                    print(f"[index] Downloaded {i}/{len(missing_vids)} (ok={dl_ok}, elapsed={time.perf_counter()-t0:.1f}s)", flush=True)

    # 2. Parallel extract missing .npy keypoints (reusing Holistic per worker process)
    to_extract = [v for v in unique_vids if _existing_npy(v) is None]
    print(f"[index] Extracting keypoints for {len(to_extract)} videos (workers={args.extract_workers})...", flush=True)
    vid_to_npy = {v: str(p) for v in unique_vids if (p := _existing_npy(v)) is not None}
    ext_err = 0
    if to_extract:
        with ProcessPoolExecutor(max_workers=args.extract_workers, initializer=_init_extract_worker) as pool:
            for i, (vid, npy_path, err) in enumerate(pool.map(_extract_one, to_extract, chunksize=6), start=1):
                if npy_path:
                    vid_to_npy[vid] = npy_path
                else:
                    ext_err += 1
                if i % 100 == 0 or i == len(to_extract):
                    print(f"[index] Extracted {i}/{len(to_extract)} (total_npy={len(vid_to_npy)}, err={ext_err}, elapsed={time.perf_counter()-t0:.1f}s)", flush=True)

    if args.extract_only:
        print(f"[index] Extract-only complete in {time.perf_counter()-t0:.1f}s (total_npy={len(vid_to_npy)}, err={ext_err})", flush=True)
        return

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
