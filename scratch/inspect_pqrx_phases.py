import json

for letter in ["P", "Q", "R", "X"]:
    path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    print(f"--- Letter {letter} (Total samples: {len(samples)}) ---")
    phases = {}
    for s in samples:
        p = s.get("phase", "none")
        phases[p] = phases.get(p, 0) + 1
    print(f"Phases: {phases}")
    hold_frames = [s for s in samples if s.get("phase") in ["stable_pose", "hold", "static"]]
    if not hold_frames:
        hold_frames = samples
    print(f"Sample frames available: {[s.get('frame_idx') for s in samples[:10]]}...")
