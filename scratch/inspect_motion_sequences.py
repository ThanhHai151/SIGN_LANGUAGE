import os
import json

seq_dir = "backend/data/alphabet_v2/motion_sequences"
letters = ["Ă", "Â", "Ê", "Đ", "Ơ", "Ư", "Ô"]

for let in letters:
    fpath = os.path.join(seq_dir, f"{let}.json")
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("sequence", [])
        video_id = data.get("video_id") or data.get("source_video")
        meta = {k: v for k, v in data.items() if k != "sequence"}
        valid_frames = sum(1 for s in seq if s.get("landmarks"))
        t_start = seq[0].get("timestamp_sec") if seq else None
        t_end = seq[-1].get("timestamp_sec") if seq else None
        print(f"Letter {let}: file={os.path.basename(fpath)}, video_id={video_id}, total_frames={len(seq)}, valid_lms={valid_frames}, time_range={t_start}..{t_end}")
        print(f"  Meta: {meta}")
    else:
        print(f"Letter {let}: Not found at {fpath}")

