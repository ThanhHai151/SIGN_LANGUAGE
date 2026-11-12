import json

fpath = "backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl"
print(f"Inspecting landmarks in {fpath}:")

total = 0
not_none = 0
valid_21 = 0
lengths = {}

with open(fpath, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        total += 1
        data = json.loads(line)
        lm = data.get("landmarks")
        if lm is not None:
            not_none += 1
            l_len = len(lm) if isinstance(lm, list) else "not_list"
            lengths[l_len] = lengths.get(l_len, 0) + 1
            if isinstance(lm, list) and len(lm) == 21:
                valid_21 += 1
                if valid_21 == 1:
                    print(f"Sample frame {idx} landmark 0:", lm[0])
                    print(f"Sample frame {idx} landmark 4:", lm[4])

print(f"Total frames: {total}")
print(f"Landmarks not None: {not_none}")
print(f"Landmarks valid 21 items: {valid_21}")
print(f"Lengths distribution: {lengths}")

