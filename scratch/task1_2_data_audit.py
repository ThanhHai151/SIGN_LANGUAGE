import json
import os
import glob
import numpy as np

def inspect_fixtures():
    fixtures = {}
    for letter in ["O", "A", "S", "C"]:
        path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
        if not os.path.exists(path):
            fixtures[letter] = {"exists": False}
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fixtures[letter] = {
            "exists": True,
            "path": path,
            "size_bytes": os.path.getsize(path),
            "type": type(data).__name__,
            "total_items": len(data) if isinstance(data, (list, dict)) else 0,
            "sample_keys": list(data[0].keys()) if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) else (list(data.keys())[:5] if isinstance(data, dict) else [])
        }
    return fixtures

def inspect_npz():
    npz_files = [
        "backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz",
        "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz",
        "backend/data/vsl_dataset.npz"
    ]
    results = {}
    for path in npz_files:
        if not os.path.exists(path):
            results[path] = {"exists": False}
            continue
        try:
            data = np.load(path, allow_pickle=True)
            results[path] = {
                "exists": True,
                "size_bytes": os.path.getsize(path),
                "files": list(data.files),
            }
            for k in data.files:
                arr = data[k]
                results[path][f"{k}_shape"] = list(arr.shape)
                results[path][f"{k}_dtype"] = str(arr.dtype)
                if k in ("y", "labels", "classes", "letter"):
                    # Check unique classes
                    try:
                        unique = np.unique(arr)
                        results[path][f"{k}_unique"] = [str(x) for x in unique[:35]]
                        results[path][f"{k}_has_O"] = bool("O" in unique or b"O" in unique or 14 in unique or 16 in unique)
                    except Exception as e:
                        results[path][f"{k}_err"] = str(e)
        except Exception as e:
            results[path] = {"exists": True, "error": str(e)}
    return results

def inspect_pilot_logs():
    pilot_dir = "backend/data/diagnostics"
    files = glob.glob(os.path.join(pilot_dir, "pilot_*.jsonl"))
    results = {}
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        o_frames = 0
        total_frames = 0
        expected_signs = set()
        trials = set()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                total_frames += 1
                try:
                    rec = json.loads(line)
                    exp = rec.get("expected_sign")
                    if exp: expected_signs.add(exp)
                    if exp == "O": o_frames += 1
                    t_id = rec.get("trial_id")
                    if t_id: trials.add(t_id)
                except Exception:
                    pass
        results[fname] = {
            "size_bytes": os.path.getsize(fpath),
            "total_frames": total_frames,
            "o_frames": o_frames,
            "expected_signs": sorted(list(expected_signs)),
            "trial_count": len(trials),
            "trials": sorted(list(trials))[:15]
        }
    return results

if __name__ == "__main__":
    report = {
        "fixtures": inspect_fixtures(),
        "npz": inspect_npz(),
        "pilot_logs": inspect_pilot_logs()
    }
    with open("scratch/task1_2_inventory_raw.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Inventory complete. Written to scratch/task1_2_inventory_raw.json")

