import os
import sys
import json
import shutil

sys.path.insert(0, ".")

orig_path = "backend/data/alphabet_v2/motion_sequences/Ô.json"
backup_path = "backend/data/alphabet_v2/motion_sequences/Ô_D0507B_legacy.json"

if os.path.exists(orig_path) and not os.path.exists(backup_path):
    shutil.copy(orig_path, backup_path)
    print(f"Backed up legacy D0507B sequence to: {backup_path}")

with open("scratch/d0507n_extracted_raw.json", encoding="utf-8") as f:
    raw_d0507n = json.load(f)

valid_cnt = sum(1 for s in raw_d0507n if s.get("quality_status") == "ok")

new_o_data = {
    "letter": "Ô",
    "type": "dynamic",
    "video_id": "D0507N",
    "region": "Miền Nam",
    "provenance": {
        "source_video": "backend/data/media_cache/videos/D0507N.mp4",
        "extractor": "MediaPipe Hands WASM (Chrome Headless v1.0.0)",
        "resolution": "1280x720",
        "fps": 23.976,
        "total_frames": len(raw_d0507n),
        "valid_landmarks": valid_cnt,
        "date_extracted": "2026-11-13",
        "convention_policy": "TOÀN QUỐC -> MIỀN NAM (QIPEDC q-o-2)"
    },
    "total_frames": len(raw_d0507n),
    "valid_frames": valid_cnt,
    "sequence": raw_d0507n
}

with open(orig_path, "w", encoding="utf-8") as f:
    json.dump(new_o_data, f, ensure_ascii=False, indent=2)

print(f"Updated {orig_path} to use D0507N ({len(raw_d0507n)} frames, {valid_cnt} valid)")

