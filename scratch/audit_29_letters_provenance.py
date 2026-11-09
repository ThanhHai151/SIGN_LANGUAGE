import os
import sys
import json
import numpy as np
import glob

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LETTERS_29 = [
    "A", "Ă", "Â", "B", "C", "D", "Đ", "E", "Ê", 
    "G", "H", "I", "K", "L", "M", "N", "O", "Ô", 
    "Ơ", "P", "Q", "R", "S", "T", "U", "Ư", "V", 
    "X", "Y"
]

BASE_22 = [
    "A", "B", "C", "D", "E", "G", "H", "I", "K", "L", 
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", 
    "X", "Y"
]

DYNAMIC_7 = ["Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"]

# 1. Load QIPEDC dictionary
with open('backend/data/sources/qipedc_dictionary.json', 'r', encoding='utf-8') as f:
    qipedc_dict = json.load(f)

# Find all entries corresponding to alphabet letters
letter_qipedc_map = {l: [] for l in LETTERS_29}
for row in qipedc_dict:
    # row format: [code, word, video_url, desc, ...]
    code = row[0]
    word = row[1].strip()
    desc = row[3] if len(row) > 3 else ""
    w_upper = word.upper()
    if w_upper in LETTERS_29:
        letter_qipedc_map[w_upper].append({
            "code": code,
            "word": word,
            "desc": desc
        })

# 2. Check local video files in media_cache
video_files = glob.glob('backend/data/media_cache/videos/*.mp4')
video_basenames = {os.path.splitext(os.path.basename(v))[0]: v for v in video_files}

# 3. Load static_landmarks JSONs
static_json_map = {}
for l in BASE_22:
    p = f'backend/data/alphabet_v2/static_landmarks/{l}.json'
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
            static_json_map[l] = d
    else:
        static_json_map[l] = None

# 4. Load motion_sequences JSONs
motion_json_map = {}
for l in DYNAMIC_7:
    p = f'backend/data/alphabet_v2/motion_sequences/{l}.json'
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
            motion_json_map[l] = d
    else:
        motion_json_map[l] = None

# 5. Load NPZ dataset
npz_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
npz_meta = json.loads(str(npz_data['metadata_json']))
npz_classes = list(npz_data['classes'])
npz_y = npz_data['y']

# Group NPZ samples by class
npz_samples_by_class = {c: [] for c in npz_classes}
for idx, item in enumerate(npz_meta):
    lbl = item['label']
    if lbl in npz_samples_by_class:
        npz_samples_by_class[lbl].append((idx, item))

# 6. Load motion dataset JSON
with open('backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json', 'r', encoding='utf-8') as f:
    motion_v2 = json.load(f)

# Analyze each letter
results = {}

for l in LETTERS_29:
    res = {
        "letter": l,
        "is_static": l in BASE_22,
        "is_dynamic": l in DYNAMIC_7,
        "qipedc_entries": letter_qipedc_map[l],
        "qipedc_codes": [e['code'] for e in letter_qipedc_map[l]],
        "available_videos": [],
        "used_in_static_json": None,
        "used_in_motion_json": None,
        "npz_samples_count": 0,
        "npz_videos": set(),
        "npz_signers": set(),
        "motion_v2_video": None,
        "motion_v2_frames_count": 0,
        "motion_v2_signers": set(),
        "south_confirmed_vids": [],
        "north_vids": [],
        "central_vids": [],
        "unknown_region_vids": [],
        "untraceable_count": 0,
        "final_classification": "UNKNOWN_REGION",
        "notes": []
    }
    
    # Check videos available on disk
    for code in res["qipedc_codes"]:
        if code in video_basenames:
            res["available_videos"].append(code)
            
    # Classify QIPEDC codes by regional suffix
    for code in res["qipedc_codes"]:
        if code.endswith('N'):
            res["south_confirmed_vids"].append(code)
        elif code.endswith('B'):
            res["north_vids"].append(code)
        elif code.endswith('T'):
            res["central_vids"].append(code)
        else:
            res["unknown_region_vids"].append(code)
            
    # Check intermediate static JSON
    if l in static_json_map and static_json_map[l] is not None:
        sj = static_json_map[l]
        vid = sj.get('video_id')
        res["used_in_static_json"] = vid
        samples = sj.get('samples', [])
        for s in samples:
            signer = s.get('signer_id')
            if signer:
                res["npz_signers"].add(signer)
                
    # Check intermediate motion JSON
    if l in motion_json_map and motion_json_map[l] is not None:
        mj = motion_json_map[l]
        vid = mj.get('video_id')
        res["used_in_motion_json"] = vid
        signer = mj.get('signer_id', 'qipedc_signer_01')
        res["motion_v2_signers"].add(signer)
        
    # Check NPZ dataset
    if l in npz_samples_by_class:
        items = npz_samples_by_class[l]
        res["npz_samples_count"] = len(items)
        for idx, item in items:
            vid = item.get('video_id')
            if vid:
                res["npz_videos"].add(vid)
            else:
                res["untraceable_count"] += 1
                
    # Check motion v2 dataset
    if l in motion_v2.get('sequences', {}):
        seq = motion_v2['sequences'][l]
        res["motion_v2_video"] = seq.get('video_id')
        res["motion_v2_frames_count"] = seq.get('total_valid_frames', 0)
        
    results[l] = res

# Print full breakdown
print(json.dumps({k: {
    "letter": v["letter"],
    "is_static": v["is_static"],
    "is_dynamic": v["is_dynamic"],
    "qipedc_codes": v["qipedc_codes"],
    "available_videos": v["available_videos"],
    "used_in_static_json": v["used_in_static_json"],
    "used_in_motion_json": v["used_in_motion_json"],
    "npz_samples_count": v["npz_samples_count"],
    "npz_videos": list(v["npz_videos"]),
    "npz_signers": list(v["npz_signers"]),
    "motion_v2_video": v["motion_v2_video"],
    "motion_v2_frames_count": v["motion_v2_frames_count"],
    "south_confirmed_vids": v["south_confirmed_vids"],
    "north_vids": v["north_vids"],
    "central_vids": v["central_vids"],
    "unknown_region_vids": v["unknown_region_vids"]
} for k, v in results.items()}, ensure_ascii=False, indent=2))

