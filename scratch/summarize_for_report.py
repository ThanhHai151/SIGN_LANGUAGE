import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

with open("backend/tests/fixtures/real_alphabet_landmarks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]
letter_stats = {}
for s in samples:
    l = s["letter"]
    if l not in letter_stats:
        letter_stats[l] = {
            "letter": l,
            "video_id": s["video_id"],
            "total": 0,
            "ok": 0,
            "no_hand": 0,
            "dev_eval_ok": 0,
            "phases": set()
        }
    st = letter_stats[l]
    st["total"] += 1
    if s["quality_status"] == "ok":
        st["ok"] += 1
        if s["split"] in ("dev", "eval"):
            st["dev_eval_ok"] += 1
    elif s["quality_status"] == "no_hand_detected":
        st["no_hand"] += 1
    st["phases"].add(s["phase"])

for k in letter_stats:
    letter_stats[k]["phases"] = sorted(list(letter_stats[k]["phases"]))

print(json.dumps(letter_stats, ensure_ascii=False, indent=2))
