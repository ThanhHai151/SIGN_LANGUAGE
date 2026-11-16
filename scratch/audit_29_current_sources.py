import os
import json

letters_29 = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
]

print(f"{'Letter':<6} | {'Video ID':<8} | {'Type':<8} | {'Total':<6} | {'Valid':<6} | {'Path'}")
print("-" * 75)

for l in letters_29:
    sm = f"backend/data/alphabet_v2/motion_sequences/{l}.json"
    ss = f"backend/data/alphabet_v2/static_landmarks/{l}.json"
    if os.path.exists(sm):
        fpath = sm
        ltype = "dynamic"
    elif os.path.exists(ss):
        fpath = ss
        ltype = "static"
    else:
        print(f"{l:<6} | MISSING!")
        continue

    with open(fpath, encoding="utf-8") as f:
        d = json.load(f)

    vid = d.get("video_id", "unknown")
    tot = d.get("total_frames", 0)
    val = d.get("valid_frames", 0)
    print(f"{l:<6} | {vid:<8} | {ltype:<8} | {tot:<6} | {val:<6} | {fpath}")

