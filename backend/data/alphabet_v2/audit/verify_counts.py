import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')

static_dir = 'backend/data/alphabet_v2/static_landmarks'
motion_dir = 'backend/data/alphabet_v2/motion_sequences'
rep_file = 'backend/data/alphabet_v2/extraction_report.json'

with open(rep_file, 'r', encoding='utf-8') as fp:
    rep = json.load(fp)

stats = rep['stats_per_letter']
print("=== 1. KIỂM TRA SỐ LIỆU TỪ extraction_report.json ===")
sum_static_total_f = 0
sum_static_valid_f = 0
sum_dyn_total_f = 0
sum_dyn_valid_f = 0

for l, info in stats.items():
    if info['type'] == 'static':
        sum_static_total_f += info['total_frames_scanned']
        sum_static_valid_f += info['valid_landmarks']
    else:
        sum_dyn_total_f += info['total_frames_scanned']
        sum_dyn_valid_f += info['valid_landmarks']

print(f"22 chữ tĩnh (trong stats_per_letter): Tổng frame = {sum_static_total_f}, Valid = {sum_static_valid_f}")
print(f"7 chữ động (trong stats_per_letter):  Tổng frame = {sum_dyn_total_f}, Valid = {sum_dyn_valid_f}")
print(f"Tổng 29 chữ (stats cộng lại):         Tổng frame = {sum_static_total_f + sum_dyn_total_f}, Valid = {sum_static_valid_f + sum_dyn_valid_f}")
print(f"Report Header ghi:                    Tổng frame = {rep['total_frames_scanned']}, Valid = {rep['total_valid_landmarks']}")

print("\n=== 2. KIỂM TRA TRỰC TIẾP TỪ CÁC FILE TRÊN ĐĨA ===")
disk_static_tot = 0
disk_static_valid = 0
per_letter_static = {}
for sf in sorted(os.listdir(static_dir)):
    with open(os.path.join(static_dir, sf), 'r', encoding='utf-8') as fp:
        d = json.load(fp)
    l = d['letter']
    s_tot = len(d['samples'])
    s_ok = sum(1 for s in d['samples'] if s.get('quality_status') == 'ok')
    disk_static_tot += s_tot
    disk_static_valid += s_ok
    per_letter_static[l] = (s_tot, s_ok)

disk_dyn_tot = 0
disk_dyn_valid = 0
per_letter_dyn = {}
for mf in sorted(os.listdir(motion_dir)):
    with open(os.path.join(motion_dir, mf), 'r', encoding='utf-8') as fp:
        d = json.load(fp)
    l = d['letter']
    m_tot = len(d['sequence'])
    m_ok = sum(1 for s in d['sequence'] if s.get('quality_status') == 'ok')
    disk_dyn_tot += m_tot
    disk_dyn_valid += m_ok
    per_letter_dyn[l] = (m_tot, m_ok)

print(f"Disk 22 chữ tĩnh: Tổng frame = {disk_static_tot}, Valid = {disk_static_valid}")
print(f"Disk 7 chữ động:  Tổng frame = {disk_dyn_tot}, Valid = {disk_dyn_valid}")
print(f"Disk Tổng 29 chữ: Tổng frame = {disk_static_tot + disk_dyn_tot}, Valid = {disk_static_valid + disk_dyn_valid}")

print("\n=== 3. XÁC MINH CHI TIẾT TỪNG CHỮ CỦA 22 CHỮ TĨNH ===")
for l, (tot, ok) in sorted(per_letter_static.items()):
    print(f"  {l}: Tổng={tot}, OK={ok}, Fail={tot-ok}")

print(f"\n=> TỔNG CỘNG 22 CHỮ TĨNH: {disk_static_valid} MẪU HỢP LỆ")

print("\n=== 4. XÁC MINH CHI TIẾT TỪNG CHỮ CỦA 7 CHỮ ĐỘNG ===")
for l, (tot, ok) in sorted(per_letter_dyn.items()):
    print(f"  {l}: Tổng={tot}, OK={ok}, Fail={tot-ok}")

print(f"\n=> TỔNG CỘNG 7 CHỮ ĐỘNG: {disk_dyn_valid} MẪU HỢP LỆ")

