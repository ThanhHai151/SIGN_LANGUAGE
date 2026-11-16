import json

for letter in ["P", "Q", "R", "X"]:
    with open(f"backend/data/alphabet_v2/static_landmarks/{letter}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\nLetter {letter}: total={len(data['samples'])}")
    phases = {}
    for i, s in enumerate(data['samples']):
        phase = s.get("phase", "none")
        phases[phase] = phases.get(phase, 0) + 1
    print("Phases:", phases)

