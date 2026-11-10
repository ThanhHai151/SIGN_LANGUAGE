import os
import shutil

QUARANTINE_BASE = os.path.abspath("backend/data/alphabet_v2/cleanup_quarantine")
TEMP_FRAMES_DIR = os.path.abspath("backend/data/alphabet_v2/temp_frames")
ANNOTATED_FRAMES_DIR = os.path.abspath("backend/data/alphabet_v2/audit/phase_review/annotated_frames")
OVERLAYS_DIR = os.path.abspath("backend/data/alphabet_v2/audit/phase_review/overlays")

print("=== STARTING CLEANUP ===")

# 1. Location 1: temp_frames
temp_deleted_count = 0
temp_deleted_bytes = 0
if os.path.exists(TEMP_FRAMES_DIR):
    for f in os.listdir(TEMP_FRAMES_DIR):
        p = os.path.join(TEMP_FRAMES_DIR, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            os.remove(p)
            temp_deleted_count += 1
            temp_deleted_bytes += sz
    print(f"[temp_frames] Deleted {temp_deleted_count} files, freed {temp_deleted_bytes / (1024*1024):.2f} MB")
else:
    print("[temp_frames] Directory does not exist!")

# 2. Location 2: annotated_frames -> Quarantine
annotated_quarantine_dir = os.path.join(QUARANTINE_BASE, "audit", "phase_review", "annotated_frames")
os.makedirs(annotated_quarantine_dir, exist_ok=True)

annotated_moved_count = 0
annotated_moved_bytes = 0
if os.path.exists(ANNOTATED_FRAMES_DIR):
    for f in os.listdir(ANNOTATED_FRAMES_DIR):
        src = os.path.join(ANNOTATED_FRAMES_DIR, f)
        dst = os.path.join(annotated_quarantine_dir, f)
        if os.path.isfile(src):
            sz = os.path.getsize(src)
            shutil.move(src, dst)
            annotated_moved_count += 1
            annotated_moved_bytes += sz
    print(f"[annotated_frames] Moved {annotated_moved_count} files to quarantine, total {annotated_moved_bytes / (1024*1024):.2f} MB")
else:
    print("[annotated_frames] Directory does not exist!")

# 3. Location 3: overlays
overlays_deleted_count = 0
overlays_deleted_bytes = 0
if os.path.exists(OVERLAYS_DIR):
    for f in os.listdir(OVERLAYS_DIR):
        p = os.path.join(OVERLAYS_DIR, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            os.remove(p)
            overlays_deleted_count += 1
            overlays_deleted_bytes += sz
    print(f"[overlays] Deleted {overlays_deleted_count} files, freed {overlays_deleted_bytes / (1024*1024):.2f} MB")
else:
    print("[overlays] Directory does not exist!")

print("=== CLEANUP EXECUTION COMPLETED ===")

