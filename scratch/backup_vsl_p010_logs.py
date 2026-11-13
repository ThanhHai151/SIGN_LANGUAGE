"""
Script sao lưu toàn diện và xác minh tính toàn vẹn của 14 session logs VSL-P010.
Chỉ thực hiện sao chép và xác minh, không xóa bất kỳ file nào.
"""
import os
import shutil
import hashlib
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS_DIR = os.path.join(PROJECT_ROOT, "debug", "words")
ARCHIVE_BASE_DIR = os.path.join(PROJECT_ROOT, "debug", "archive")
ARCHIVE_TARGET_DIR = os.path.join(ARCHIVE_BASE_DIR, "vsl_p010_historical_reconstructed_2026-11-07")

os.makedirs(ARCHIVE_TARGET_DIR, exist_ok=True)

# 14 session chuẩn từ 2026-11-07
EXPECTED_SESSIONS = [
    "Ban_trial_01_2026-11-07_150156",
    "Ban_trial_01_2026-11-07_151940",
    "Ban_trial_01_2026-11-07_161034",
    "Ban_trial_02_2026-11-07_150513",
    "Ban_trial_02_2026-11-07_161126",
    "Ten_la_gi_trial_01_2026-11-07_145001",
    "Ten_la_gi_trial_01_2026-11-07_161207",
    "Ten_la_gi_trial_02_2026-11-07_150839",
    "Ten_la_gi_trial_02_2026-11-07_161709",
    "The_nao_trial_01_2026-11-07_145112",
    "The_nao_trial_01_2026-11-07_150725",
    "The_nao_trial_01_2026-11-07_161800",
    "The_nao_trial_02_2026-11-07_150758",
    "The_nao_trial_02_2026-11-07_161857"
]

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("=" * 70)
print(f"BẮT ĐẦU SAO LƯU 14 SESSIONS TỪ {WORDS_DIR} -> {ARCHIVE_TARGET_DIR}")
print("=" * 70)

manifest = {
    "archive_name": "vsl_p010_historical_reconstructed_2026-11-07",
    "source_dir": WORDS_DIR,
    "destination_dir": ARCHIVE_TARGET_DIR,
    "total_sessions": len(EXPECTED_SESSIONS),
    "sessions": {},
    "files": {}
}

total_copied_files = 0
total_bytes = 0

# 1. Sao chép từng session
for sess_name in EXPECTED_SESSIONS:
    src_sess_dir = os.path.join(WORDS_DIR, sess_name)
    dst_sess_dir = os.path.join(ARCHIVE_TARGET_DIR, sess_name)
    if not os.path.exists(src_sess_dir):
        print(f"[ERROR] Không tìm thấy session: {src_sess_dir}")
        continue
    
    os.makedirs(dst_sess_dir, exist_ok=True)
    sess_manifest = {"session_name": sess_name, "files_count": 0, "total_bytes": 0}

    for root, dirs, files in os.walk(src_sess_dir):
        rel_path = os.path.relpath(root, src_sess_dir)
        target_sub_dir = os.path.join(dst_sess_dir, rel_path) if rel_path != "." else dst_sess_dir
        os.makedirs(target_sub_dir, exist_ok=True)

        for filename in files:
            src_file = os.path.join(root, filename)
            dst_file = os.path.join(target_sub_dir, filename)

            # Sao chép bảo lưu metadata
            shutil.copy2(src_file, dst_file)

            # Kiểm tra kích thước và checksum
            src_size = os.path.getsize(src_file)
            dst_size = os.path.getsize(dst_file)
            assert src_size == dst_size, f"Size mismatch for {filename}: {src_size} vs {dst_size}"

            src_hash = sha256_file(src_file)
            dst_hash = sha256_file(dst_file)
            assert src_hash == dst_hash, f"Hash mismatch for {filename}"

            total_copied_files += 1
            total_bytes += src_size
            sess_manifest["files_count"] += 1
            sess_manifest["total_bytes"] += src_size

            rel_file_path = os.path.relpath(dst_file, ARCHIVE_TARGET_DIR).replace("\\", "/")
            manifest["files"][rel_file_path] = {
                "size_bytes": dst_size,
                "sha256": dst_hash
            }

    manifest["sessions"][sess_name] = sess_manifest
    print(f"  [OK] Đã sao lưu & xác minh session: {sess_name} ({sess_manifest['files_count']} files, {sess_manifest['total_bytes']:,} bytes)")

# 2. Sao chép và làm sạch trials_summary.csv cho archive (chỉ giữ đúng 14 session gốc)
src_master_csv = os.path.join(WORDS_DIR, "trials_summary.csv")
dst_master_csv = os.path.join(ARCHIVE_TARGET_DIR, "trials_summary.csv")
if os.path.exists(src_master_csv):
    with open(src_master_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Header + 14 dòng session
    header = lines[0]
    pristine_rows = [header]
    for line in lines[1:]:
        sess_id = line.split(",")[0]
        if sess_id in EXPECTED_SESSIONS:
            pristine_rows.append(line)
    
    with open(dst_master_csv, "w", encoding="utf-8") as f:
        f.writelines(pristine_rows)
    
    total_copied_files += 1
    csv_size = os.path.getsize(dst_master_csv)
    total_bytes += csv_size
    manifest["files"]["trials_summary.csv"] = {
        "size_bytes": csv_size,
        "sha256": sha256_file(dst_master_csv),
        "total_rows": len(pristine_rows) - 1
    }
    print(f"  [OK] Đã tạo trials_summary.csv chuẩn 14 dòng cho archive ({csv_size:,} bytes)")

manifest["total_copied_files"] = total_copied_files
manifest["total_bytes"] = total_bytes

# Ghi manifest
manifest_path = os.path.join(ARCHIVE_TARGET_DIR, "ARCHIVE_MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("=" * 70)
print(f"HOÀN THÀNH SAO LƯU: {total_copied_files} files, {total_bytes:,} bytes (~{total_bytes/(1024*1024):.2f} MB)")
print(f"Manifest đã lưu tại: {manifest_path}")
print("=" * 70)

