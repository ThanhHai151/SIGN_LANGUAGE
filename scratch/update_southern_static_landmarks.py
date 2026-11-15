import os
import json
import shutil

STATIC_DIR = "backend/data/alphabet_v2/static_landmarks"

updates = [
    ("G", "D0498B", "D0498N", "scratch/d0498n_extracted_raw.json"),
    ("P", "D0509B", "D0509N", "scratch/d0509n_extracted_raw.json"),
    ("T", "D0513B", "D0513N", "scratch/d0513n_extracted_raw.json"),
    ("X", "D0517B", "D0517N", "scratch/d0517n_extracted_raw.json"),
]

# 1. Update G, P, T, X
for letter, old_vid, new_vid, raw_path in updates:
    curr_path = os.path.join(STATIC_DIR, f"{letter}.json")
    legacy_path = os.path.join(STATIC_DIR, f"{letter}_{old_vid}_legacy.json")
    
    if os.path.exists(curr_path) and not os.path.exists(legacy_path):
        shutil.copy(curr_path, legacy_path)
        print(f"Backed up legacy {curr_path} -> {legacy_path}")
        
    with open(raw_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    valid_cnt = sum(1 for s in samples if s.get("quality_status") == "ok")
    
    new_data = {
        "letter": letter,
        "type": "static_landmarks",
        "video_id": new_vid,
        "region": "Miền Nam",
        "provenance": {
            "source_video": f"backend/data/media_cache/videos/{new_vid}.mp4",
            "extractor": "MediaPipe Hands WASM (Chrome Headless v1.0.0)",
            "convention_policy": "TOÀN QUỐC -> MIỀN NAM (QIPEDC)",
            "legacy_preserved": legacy_path
        },
        "total_frames": len(samples),
        "valid_frames": valid_cnt,
        "samples": samples
    }
    
    with open(curr_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"Updated {curr_path}: video_id={new_vid}, total={len(samples)}, valid={valid_cnt}")

# 2. Update H from D0500N.json
curr_h_path = os.path.join(STATIC_DIR, "H.json")
legacy_h_path = os.path.join(STATIC_DIR, "H_D0500B_legacy.json")
d0500n_path = os.path.join(STATIC_DIR, "D0500N.json")

if os.path.exists(curr_h_path) and not os.path.exists(legacy_h_path):
    shutil.copy(curr_h_path, legacy_h_path)
    print(f"Backed up legacy {curr_h_path} -> {legacy_h_path}")

with open(d0500n_path, "r", encoding="utf-8") as f:
    d0500n_data = json.load(f)

h_samples = d0500n_data.get("samples", [])
h_valid = sum(1 for s in h_samples if s.get("quality_status") == "ok")

new_h_data = {
    "letter": "H",
    "type": "static_landmarks",
    "video_id": "D0500N",
    "region": "Miền Nam",
    "provenance": {
        "source_video": "backend/data/media_cache/videos/D0500N.mp4",
        "extractor": "MediaPipe Hands WASM",
        "convention_policy": "TOÀN QUỐC -> MIỀN NAM (QIPEDC)",
        "legacy_preserved": legacy_h_path
    },
    "total_frames": len(h_samples),
    "valid_frames": h_valid,
    "samples": h_samples
}

with open(curr_h_path, "w", encoding="utf-8") as f:
    json.dump(new_h_data, f, ensure_ascii=False, indent=2)
print(f"Updated {curr_h_path}: video_id=D0500N, total={len(h_samples)}, valid={h_valid}")

