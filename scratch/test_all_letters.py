import sys
sys.path.insert(0, ".")
import json
import glob
from backend.vision.alphabet_classifier import alphabet_classifier

all_files = glob.glob("backend/data/alphabet_v2/static_landmarks/*.json")

correct = {}
false_positives = {}

for f in all_files:
    data = json.load(open(f, encoding="utf-8"))
    let = data.get("letter")
    samples = [s for s in data.get("samples", []) if s.get("phase") == "stable_pose" and s.get("landmarks") is not None]
    for s in samples:
        lm = s["landmarks"]
        res = alphabet_classifier.classify_detailed(lm)
        pred = res.get("letter")
        status = res.get("status")
        
        # Check I classification
        if pred == "I":
            if let == "I":
                correct["I"] = correct.get("I", 0) + 1
            else:
                false_positives[let] = false_positives.get(let, 0) + 1

print("Currently correct I:", correct.get("I", 0), "out of", len([s for f in all_files for s in json.load(open(f, encoding="utf-8")).get("samples", []) if json.load(open(f, encoding="utf-8")).get("letter") == "I" and s.get("phase") == "stable_pose" and s.get("landmarks") is not None]))
print("False positives as I on other letters:", false_positives)

