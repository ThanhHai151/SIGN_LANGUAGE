import json

for letter in ["P", "Q", "R", "X"]:
    path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"--- Letter {letter} ---")
    print(f"Keys: {list(data.keys())}")
    print(f"Video ID: {data.get('video_id')}, Letter: {data.get('letter')}, Total frames: {data.get('total_frames')}")
    stable_frames = data.get("stable_frames", [])
    print(f"Stable frames count: {len(stable_frames)}")
    if stable_frames:
        print(f"First stable frame: idx={stable_frames[0].get('frame_idx')}, phase={stable_frames[0].get('phase')}")
        print(f"Stable frame indices: {[f.get('frame_idx') for f in stable_frames]}")
