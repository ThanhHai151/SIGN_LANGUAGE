import json, os, glob
import numpy as np

print("=== CHECKING DATASET AND MAPPING FOR LETTER S ===")

# 1. Dataset static v2
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
npz = np.load(npz_path, allow_pickle=True)
classes = npz["classes"]
y = npz["y"]
X = npz["X"]
lms = npz["landmarks"]
meta = json.loads(str(npz["metadata_json"]))

s_class_idx = int(np.where(classes == "S")[0][0])
s_mask = (y == s_class_idx)
s_samples_count = int(np.sum(s_mask))
s_landmarks = lms[s_mask]
s_X = X[s_mask]

print(f"Dataset path: {npz_path}")
print(f"Classes: {classes.tolist()}")
print(f"Letter S label index: {s_class_idx}")
print(f"Total S samples in vsl_alphabet_static_v2.npz: {s_samples_count}")
print(f"Landmarks shape for S: {s_landmarks.shape}")
print(f"Feature vector X shape for S: {s_X.shape}")

# 2. Check metadata for S in dataset
s_meta = [m for m in meta if m.get("label") == "S"]
print(f"Total S entries in metadata_json: {len(s_meta)}")
for i, m in enumerate(s_meta):
    print(f"  Sample {i:2d}: frame={m.get('frame_index')}, video_id={m.get('video_id')}, source_video={m.get('source_video')}, is_rep={m.get('is_representative')}, quality={m.get('quality_status')}")

# 3. Check static_landmarks S.json
s_json_path = "backend/data/alphabet_v2/static_landmarks/S.json"
with open(s_json_path, "r", encoding="utf-8") as f:
    s_json = json.load(f)

print(f"\nStatic landmark file: {s_json_path}")
print(f"Letter: {s_json.get('letter')}, type: {s_json.get('type')}, video_id: {s_json.get('video_id')}")
print(f"Total frames: {s_json.get('total_frames')}, valid frames: {s_json.get('valid_frames')}")

# 4. Check video file on disk
video_path = "backend/data/media_cache/videos/D0512.mp4"
exists = os.path.exists(video_path)
size = os.path.getsize(video_path) if exists else 0
print(f"\nSource video on disk: {video_path}")
print(f"Exists: {exists}, File size: {size} bytes ({size / (1024*1024):.2f} MB)")

# 5. Check QIPEDC lexicon entry
lexicon_path = "backend/data/vsl_lexicon.json"
with open(lexicon_path, "r", encoding="utf-8") as f:
    lexicon = json.load(f)

s_entries = [e for e in lexicon["entries"] if e.get("word") == "s" or e.get("id") == "q-s"]
print(f"\nLexicon entries for S: {len(s_entries)}")
for e in s_entries:
    print(f"  ID: {e.get('id')}, Word: {e.get('word')}, How_to: {e.get('how_to')}")
    print(f"  Video URL: {e.get('video_url')}, Image URL: {e.get('image_url')}")

# 6. Check extraction script run_alphabet_v2_extraction.py
extr_script = "backend/training/run_alphabet_v2_extraction.py"
if os.path.exists(extr_script):
    with open(extr_script, "r", encoding="utf-8") as f:
        extr_content = f.read()
    print(f"\nChecking {extr_script}:")
    for line in extr_content.splitlines():
        if "D0512" in line or '"S"' in line:
            print(f"  {line.strip()}")

