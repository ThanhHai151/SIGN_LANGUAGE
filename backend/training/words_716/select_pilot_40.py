import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
META_PATH = ROOT / "backend" / "data" / "words_716" / "sample_metadata.jsonl"
OUT_PATH = ROOT / "backend" / "data" / "words_716" / "pilot_40_classes.json"

rows = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

# Filter for clean samples (no quality flags, perfect pose, hand coverage > 0.70)
clean_rows = []
for r in rows:
    if len(r.get("quality_flags", [])) == 0 and r["pose_coverage"] >= 0.99 and not r["needs_manual_review"]:
        clean_rows.append(r)

print(f"Total clean candidates (0 quality flags, no manual review): {len(clean_rows)}")

# Define target semantic topics / communication interest keywords to prioritize
priority_words = {
    # Greetings, Questions, Pronouns
    "ai", "sao", "là gì?", "bao nhiêu?", "lúc nào", "ở đây",
    # Actions & Verbs
    "đi", "vào", "nghĩ", "muốn", "làm việc", "lấy", "chạy", "gặp", "nói", "nghe", "nhìn", "ăn", "uống", "học", "chơi", "giúp đỡ", "bảo vệ", "liên lạc", "lên xe", "qua lại",
    # Objects & Entities
    "nhà", "lớp", "bàn tay", "bàn chân", "tiền", "sách", "xe", "áo", "cửa", "buổi sáng", "buổi tối", "hoa",
    # Emotions & Descriptors
    "vui vẻ", "lo sợ", "lạc quan", "tự lập", "đẹp", "tốt", "nhanh", "chậm", "khác", "giống nhau"
}

# Score each candidate by communication relevance and quality
def score_candidate(r):
    w = r["word"].lower()
    score = 0
    if w in priority_words:
        score += 50
    for pw in priority_words:
        if pw in w:
            score += 20
    # Prefer higher hand coverage
    lh = r["left_hand_coverage"] if r["left_hand_participating"] else 1.0
    rh = r["right_hand_coverage"] if r["right_hand_participating"] else 1.0
    score += (lh + rh) * 10
    return score

clean_rows.sort(key=score_candidate, reverse=True)

# Stratified selection with fixed seed
rng = random.Random(42)

# Buckets:
# Region: NATIONAL (need ~28), SOUTH (need ~12)
# Hand: 1-hand (need ~12), 2-hand (need ~28)
# Duration: Short (<1.8s), Medium (1.8-2.5s), Long (>2.5s)

selected = []
selected_eids = set()

# Helper to pick best available satisfying predicate
def pick_one(predicate):
    for r in clean_rows:
        if r["entry_id"] not in selected_eids and predicate(r):
            selected.append(r)
            selected_eids.add(r["entry_id"])
            return True
    return False

# 1. First pick top South candidates to guarantee ~12 South
# South 1-hand
for _ in range(4):
    pick_one(lambda r: r["selected_region"] == "SOUTH" and (r["left_hand_participating"] ^ r["right_hand_participating"]))
# South 2-hand short/medium
for _ in range(4):
    pick_one(lambda r: r["selected_region"] == "SOUTH" and r["left_hand_participating"] and r["right_hand_participating"] and r["active_duration_sec"] <= 2.2)
# South 2-hand long
for _ in range(4):
    pick_one(lambda r: r["selected_region"] == "SOUTH" and r["left_hand_participating"] and r["right_hand_participating"] and r["active_duration_sec"] > 2.2)

# 2. Pick National candidates (~28)
# National 1-hand
for _ in range(8):
    pick_one(lambda r: r["selected_region"] == "NATIONAL" and (r["left_hand_participating"] ^ r["right_hand_participating"]))
# National 2-hand short (< 1.8s)
for _ in range(6):
    pick_one(lambda r: r["selected_region"] == "NATIONAL" and r["left_hand_participating"] and r["right_hand_participating"] and r["active_duration_sec"] < 1.8)
# National 2-hand medium (1.8s - 2.5s)
for _ in range(8):
    pick_one(lambda r: r["selected_region"] == "NATIONAL" and r["left_hand_participating"] and r["right_hand_participating"] and 1.8 <= r["active_duration_sec"] <= 2.5)
# National 2-hand long (> 2.5s)
for _ in range(6):
    pick_one(lambda r: r["selected_region"] == "NATIONAL" and r["left_hand_participating"] and r["right_hand_participating"] and r["active_duration_sec"] > 2.5)

# Fill to exactly 40 if needed
while len(selected) < 40:
    pick_one(lambda r: True)

# Sort selected by original class_index to preserve order
selected.sort(key=lambda r: r["class_index"])

# Validate 40 items
assert len(selected) == 40
assert len(set(r["class_index"] for r in selected)) == 40
assert len(set(r["entry_id"] for r in selected)) == 40

nat_count = sum(1 for r in selected if r["selected_region"] == "NATIONAL")
sou_count = sum(1 for r in selected if r["selected_region"] == "SOUTH")
one_count = sum(1 for r in selected if (r["left_hand_participating"] ^ r["right_hand_participating"]))
two_count = sum(1 for r in selected if (r["left_hand_participating"] and r["right_hand_participating"]))
short_count = sum(1 for r in selected if r["active_duration_sec"] < 1.8)
med_count = sum(1 for r in selected if 1.8 <= r["active_duration_sec"] <= 2.5)
long_count = sum(1 for r in selected if r["active_duration_sec"] > 2.5)

print("\n=== PILOT 40 SELECTION SUMMARY ===")
print(f"Total selected: {len(selected)}")
print(f"Region: NATIONAL = {nat_count}, SOUTH = {sou_count}")
print(f"Hands: 1-hand = {one_count}, 2-hand = {two_count}")
print(f"Duration: Short (<1.8s) = {short_count}, Medium = {med_count}, Long (>2.5s) = {long_count}")

pilot_data = {
    "schema_version": "3.2a-pilot-40-classes-v1",
    "seed": 42,
    "total_classes_in_model": 716,
    "pilot_classes_count": 40,
    "distribution": {
        "national_count": nat_count,
        "south_count": sou_count,
        "one_hand_count": one_count,
        "two_hands_count": two_count,
        "short_duration_count": short_count,
        "medium_duration_count": med_count,
        "long_duration_count": long_count,
    },
    "pilot_class_indices": [r["class_index"] for r in selected],
    "classes": [
        {
            "pilot_row": idx,
            "class_index": r["class_index"],
            "entry_id": r["entry_id"],
            "word": r["word"],
            "selected_region": r["selected_region"],
            "source_video_id": r["source_video_id"],
            "active_duration_sec": r["active_duration_sec"],
            "left_hand_participating": r["left_hand_participating"],
            "right_hand_participating": r["right_hand_participating"],
            "pose_coverage": r["pose_coverage"],
            "left_hand_coverage": r["left_hand_coverage"],
            "right_hand_coverage": r["right_hand_coverage"],
        }
        for idx, r in enumerate(selected)
    ]
}

OUT_PATH.write_text(json.dumps(pilot_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Saved pilot 40 classes to: {OUT_PATH}")
