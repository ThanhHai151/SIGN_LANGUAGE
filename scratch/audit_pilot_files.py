import os
import glob
import json
import hashlib
import time

diag_dir = os.path.abspath("backend/data/diagnostics")
files = glob.glob(os.path.join(diag_dir, "*.json*")) + glob.glob(os.path.join(diag_dir, "pilot", "*.json*"))

print("=== DIAGNOSTICS DIRECTORY INVENTORY ===")
for fpath in sorted(files):
    size = os.path.getsize(fpath)
    mtime = time.ctime(os.path.getmtime(fpath))
    print(f"\nFile: {os.path.basename(fpath)}")
    print(f"  Path: {fpath}")
    print(f"  Size: {size:,} bytes")
    print(f"  Modified: {mtime}")

    # Inspect first few lines and count
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line.strip():
                print("  Status: Empty file")
                continue
            first_rec = json.loads(first_line)
            print(f"  First record session_id: {first_rec.get('session_id')}")
            print(f"  First record trial_id: {first_rec.get('trial_id')}")
            print(f"  First record timestamp: {first_rec.get('timestamp')}")
            print(f"  PyTorch version: {first_rec.get('static_head', {}).get('pytorch_version')}")
            print(f"  has_static_head: {first_rec.get('has_static_head')}")
    except Exception as e:
        print(f"  Error reading header: {e}")

