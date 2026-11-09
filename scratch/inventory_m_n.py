"""
Kiểm kê toàn bộ video, file landmark, metadata liên quan đến chữ M và N trong kho dữ liệu.
"""
import os
import glob
import json

WORKSPACE = "d:/SIGN_LANGUAGE"

print("=== 1. TÌM KIẾM VIDEO LIÊN QUAN ĐẾN M VÀ N ===")
# Tìm trong media_cache và toàn bộ project
video_matches = []
for root, _, files in os.walk(WORKSPACE):
    if any(skip in root for skip in [".git", "node_modules", ".venv", "dist"]):
        continue
    for f in files:
        if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Check if filename contains D0504, D0505, or M, N
            low = f.lower()
            if any(k in low for k in ["d0504", "d0505", "letter_m", "letter_n", "chu_m", "chu_n"]):
                video_matches.append(os.path.join(root, f))
            # Also check all D05*.mp4 to see what exists around D0504 and D0505
            if f.startswith("D0504") or f.startswith("D0505"):
                video_matches.append(os.path.join(root, f))

video_matches = sorted(list(set(video_matches)))
for vm in video_matches:
    print(f"  - {os.path.relpath(vm, WORKSPACE)} (Size: {os.path.getsize(vm)} bytes)")

print("\n=== 2. KIỂM TRA TỪ ĐIỂN QIPEDC (qipedc_dictionary.json) ===")
qipedc_path = os.path.join(WORKSPACE, "backend/data/sources/qipedc_dictionary.json")
if os.path.exists(qipedc_path):
    with open(qipedc_path, "r", encoding="utf-8") as fp:
        q_dict = json.load(fp)
    print(f"Tổng số mục trong qipedc_dictionary: {len(q_dict)}")
    for item in q_dict:
        w = item.get("word", "").strip().upper()
        sid = item.get("id", "")
        # Check if item corresponds to M or N
        if w in ["M", "N", "CHỮ M", "CHỮ N", "CHU M", "CHU N"] or sid in ["D0504", "D0505"]:
            print(f"  Mục: word='{item.get('word')}', id='{sid}', video='{item.get('video')}', region='{item.get('region')}', dialect='{item.get('dialect')}', desc='{item.get('description')}'")
        # Check all entries where id starts with D0504 or D0505
        if str(sid).startswith("D0504") or str(sid).startswith("D0505"):
            print(f"  Match ID: {item}")

print("\n=== 3. KIỂM TRA VIDEOS METADATA (videos_metadata.json) ===")
meta_path = os.path.join(WORKSPACE, "backend/data/alphabet_v2/metadata/videos_metadata.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as fp:
        v_meta = json.load(fp)
    for k in ["M", "N"]:
        print(f"  Key '{k}': {v_meta.get(k)}")

print("\n=== 4. KIỂM TRA STATIC LANDMARKS JSON ===")
for ch in ["M", "N"]:
    sf = os.path.join(WORKSPACE, f"backend/data/alphabet_v2/static_landmarks/{ch}.json")
    if os.path.exists(sf):
        with open(sf, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        samples = d.get("samples", [])
        stable = [s for s in samples if s.get("phase") == "stable_pose"]
        vid_ids = list(set(s.get("video_id") for s in samples))
        print(f"  {ch}.json: {len(samples)} samples, {len(stable)} stable_pose, video_ids={vid_ids}")
    else:
        print(f"  {ch}.json: KHÔNG TỒN TẠI")

# Check if there are other json files in static_landmarks or motion_sequences
for root, _, files in os.walk(os.path.join(WORKSPACE, "backend/data")):
    for f in files:
        if any(k in f for k in ["D0504", "D0505", "M.json", "N.json"]):
            print(f"  Data file found: {os.path.relpath(os.path.join(root, f), WORKSPACE)}")

