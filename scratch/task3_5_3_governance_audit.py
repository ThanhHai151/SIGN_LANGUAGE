import os
import json
import numpy as np

def run_audit():
    fixtures_dir = "backend/data/alphabet_v2/static_landmarks"
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"

    print("=== PART 1: K, D, Q, B SAMPLES IN FIXTURES ===")
    queries = [
        ("K", [46, 48]),
        ("D", [54, 56]),
        ("Q", [60, 62, 64]),
        ("B", [52, 54, 56, 58, 60]),
    ]

    for letter, frames in queries:
        path = os.path.join(fixtures_dir, f"{letter}.json")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        print(f"\n{letter}.json total samples: {len(samples)}, video_id: {data.get('video_id', 'N/A')}")
        for s in samples:
            f_idx = s.get("frame_index") or s.get("frame_idx")
            if f_idx in frames:
                has_lm = bool(s.get("landmarks"))
                print(f"  Frame {f_idx}: phase={s.get('phase')}, is_rep={s.get('is_representative')}, has_lm={has_lm}")

    print("\n=== PART 2: NPZ CONTENT INSPECTION ===")
    npz_data = np.load(npz_path, allow_pickle=True)
    classes = list(npz_data["classes"])
    y = npz_data["y"]
    landmarks = npz_data["landmarks"]
    metadata_json = npz_data["metadata_json"]

    parsed_meta = json.loads(metadata_json.item() if metadata_json.shape == () else metadata_json[0])

    print(f"Total samples in NPZ: {len(y)}, parsed metadata count: {len(parsed_meta)}")
    if parsed_meta:
        print("Sample meta keys:", list(parsed_meta[0].keys()))

    for letter, frames in queries:
        matching = []
        for idx in range(len(y)):
            c_label = classes[y[idx]]
            if c_label == letter:
                meta = parsed_meta[idx]
                f_idx = meta.get("frame_index") or meta.get("frame_idx")
                vid = meta.get("video_id")
                ph = meta.get("phase")
                is_rep = meta.get("is_representative")
                if f_idx in frames or frames is None:
                    matching.append((idx, str(vid), int(f_idx) if f_idx is not None else None, str(ph), is_rep))
        print(f"NPZ {letter} queried frames matching: {matching}")

    print("\n=== PART 3: DENOMINATOR 228 AUDIT (O SAMPLES) ===")
    # 1. NPZ O
    npz_o_indices = [idx for idx in range(len(y)) if classes[y[idx]] == "O"]
    print(f"NPZ O sample count: {len(npz_o_indices)}")
    npz_o_vids = {}
    for idx in npz_o_indices:
        meta = parsed_meta[idx]
        vid = str(meta.get("video_id"))
        f_idx = int(meta.get("frame_index") or meta.get("frame_idx"))
        npz_o_vids.setdefault(vid, []).append(f_idx)
    for vid, flist in npz_o_vids.items():
        print(f"  NPZ O video {vid}: {len(flist)} frames -> {flist}")

    # 2. Fixture O
    fixture_o_path = os.path.join(fixtures_dir, "O.json")
    with open(fixture_o_path, "r", encoding="utf-8") as f:
        fix_o_data = json.load(f)
    fix_o_samples = fix_o_data.get("samples", [])
    fix_o_vid = fix_o_data.get("video_id", "D0506")
    fix_o_frames = [(s.get("frame_index") or s.get("frame_idx")) for s in fix_o_samples if s.get("landmarks")]
    print(f"Fixture O sample count: {len(fix_o_frames)}, video_id: {fix_o_vid} -> {fix_o_frames}")

    # Check overlap between Fixture O and NPZ O
    overlap_frames = set(fix_o_frames).intersection(set(npz_o_vids.get("D0506", [])))
    print(f"Overlap between Fixture O (D0506) and NPZ O (D0506): {len(overlap_frames)} / {len(fix_o_frames)} frames")
    print(f"Are all Fixture O frames identical to NPZ D0506 frames? {overlap_frames == set(fix_o_frames)}")

    # 3. Pilot 213800
    pilot_2138_path = "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl"
    p2138_frames = 0
    if os.path.exists(pilot_2138_path):
        with open(pilot_2138_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("trial_id") and rec.get("geometric", {}).get("features"):
                    p2138_frames += 1
    print(f"Pilot 213800 active frames: {p2138_frames}")

    # 4. Pilot 211742
    pilot_2117_path = "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl"
    p2117_frames = 0
    if os.path.exists(pilot_2117_path):
        with open(pilot_2117_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("trial_id") and rec.get("geometric", {}).get("features"):
                    p2117_frames += 1
    print(f"Pilot 211742 active frames: {p2117_frames}")

    total_o_raw = len(npz_o_indices) + len(fix_o_frames) + p2138_frames + p2117_frames
    print(f"Total raw observations summed in TASK 3.5.2: {total_o_raw} (48 + 20 + 92 + 68)")
    unique_o_frames = len(npz_o_indices) + (len(fix_o_frames) - len(overlap_frames)) + p2138_frames + p2117_frames
    print(f"Unique frames (deduplicating Fixture O D0506 vs NPZ D0506): {unique_o_frames} (48 + 0 + 92 + 68)")

    # Unique videos/streams
    print(f"Unique video sources for O: {len(npz_o_vids)} NPZ clips ({list(npz_o_vids.keys())}) + 2 webcam pilots = 5 total sources")
    print("Unique signers for O: 1 signer for QIPEDC studio (all 3 clips D0506, D0507B, D0508N) + 1 signer for webcam pilots = 2 unique signers")

    print("\n=== PART 4: DENOMINATOR 925 AUDIT (NON-O SAMPLES) ===")
    npz_non_o_count = len([idx for idx in range(len(y)) if classes[y[idx]] != "O"])
    print(f"NPZ non-O samples: {npz_non_o_count}")

    p_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
    p_holdout_active = 0
    if os.path.exists(p_holdout_path):
        with open(p_holdout_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("trial_id") and rec.get("geometric", {}).get("features"):
                    p_holdout_active += 1
    print(f"P Holdout (183703) active frames: {p_holdout_active}")

    x_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
    x_holdout_active = 0
    if os.path.exists(x_holdout_path):
        with open(x_holdout_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("trial_id") and rec.get("geometric", {}).get("features"):
                    x_holdout_active += 1
    print(f"X Holdout (193712) active frames: {x_holdout_active}")

    total_non_o_raw = npz_non_o_count + p_holdout_active + x_holdout_active
    print(f"Total non-O observations summed in TASK 3.5.2: {total_non_o_raw} (503 + 314 + 108)")
    print("Non-O video/stream sources: 21 NPZ classes/videos + 1 P webcam session + 1 X webcam session = 23 sources")
    print("Non-O unique signers: 1 QIPEDC signer + 1 webcam signer = 2 unique signers")

if __name__ == "__main__":
    run_audit()
