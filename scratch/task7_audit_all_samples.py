import os
import sys
import json
import numpy as np
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load NPZ
npz_path = 'backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz'
data = np.load(npz_path, allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])

print(f"Total samples in NPZ: {len(meta)}")
print(f"Total classes: {len(classes)} -> {classes}")

# Audit each sample
audited_samples = []
for idx, s in enumerate(meta):
    label = s['label']
    src_type = s.get('source_type', 'static_video')
    src_letter = s.get('source_letter', label)
    vid = s['video_id']
    fidx = s['frame_index']
    ts = s.get('timestamp_sec', 0.0)
    st = s.get('quality_status', 'unknown')
    disp = s.get('displacement_to_prev', 0.0)
    is_rep = s.get('is_representative', True)
    
    # 1. Axis A: Region
    if vid.endswith('N'):
        region = 'REGION_SOUTH'
    elif vid.endswith('B'):
        region = 'REGION_NORTH'
    elif vid.endswith('T'):
        region = 'REGION_CENTRAL'
    else:
        region = 'REGION_SHARED'
        
    # 2. Axis B: Semantic
    if src_type == 'static_video':
        # Direct static video for the assigned letter
        semantic = 'SEMANTIC_CONFIRMED'
        semantic_reason = f"Trích xuất trực tiếp từ video từ điển QIPEDC chuẩn cho chữ {label} ({vid})"
    elif src_type == 'base_pose_of_dynamic':
        # Derived from dynamic letter
        semantic = 'SEMANTIC_UNCONFIRMED'
        semantic_reason = f"Trích xuất từ pha BASE_POSE của cử chỉ động {src_letter} ({vid}), chưa có chứng nhận ngôn ngữ học độc lập tương đương chữ {label}"
    else:
        semantic = 'SEMANTIC_UNCONFIRMED'
        semantic_reason = "Không rõ nguồn gốc loại dữ liệu"

    # 3. Overall Eligibility
    if region in ('REGION_NORTH', 'REGION_CENTRAL'):
        eligibility = 'EXCLUDED_REGION'
        elig_reason = f"Nguồn gốc vùng miền là {region} ({vid}), không đưa vào dataset chuẩn miền Nam"
    elif semantic == 'SEMANTIC_UNCONFIRMED':
        eligibility = 'PENDING_VERIFICATION'
        elig_reason = f"Nguồn vùng miền hợp lệ ({region}) nhưng ý nghĩa nhãn chưa được chuyên gia VSL xác nhận ({semantic_reason})"
    elif region == 'REGION_SHARED' and semantic == 'SEMANTIC_CONFIRMED':
        eligibility = 'ELIGIBLE_SHARED'
        elig_reason = "Ký hiệu dùng chung toàn quốc (Ưu tiên 1) và ý nghĩa nhãn đã được xác minh"
    elif region == 'REGION_SOUTH' and semantic == 'SEMANTIC_CONFIRMED':
        eligibility = 'ELIGIBLE_SOUTH'
        elig_reason = "Ký hiệu riêng miền Nam (Ưu tiên 2) và ý nghĩa nhãn đã được xác minh"
    else:
        eligibility = 'PENDING_VERIFICATION'
        elig_reason = "Chưa đủ điều kiện kết luận"

    audited_samples.append({
        'sample_idx': idx,
        'label': label,
        'source_type': src_type,
        'source_letter': src_letter,
        'video_id': vid,
        'frame_index': fidx,
        'timestamp_sec': ts,
        'quality_status': st,
        'displacement_to_prev': disp,
        'is_representative': is_rep,
        'region': region,
        'semantic': semantic,
        'semantic_reason': semantic_reason,
        'eligibility': eligibility,
        'elig_reason': elig_reason
    })

# Summary statistics
print("\n" + "="*80)
print("TỔNG HỢP THEO LOẠI NGUỒN DỮ LIỆU:")
type_counts = defaultdict(int)
for s in audited_samples:
    type_counts[s['source_type']] += 1
for k, v in type_counts.items():
    print(f"  {k:25s}: {v:3d} mẫu ({v/len(audited_samples)*100:.1f}%)")

print("\n" + "="*80)
print("TỔNG HỢP THEO TRỤC VÙNG MIỀN (AXIS A):")
reg_counts = defaultdict(int)
for s in audited_samples:
    reg_counts[s['region']] += 1
for k, v in reg_counts.items():
    print(f"  {k:20s}: {v:3d} mẫu ({v/len(audited_samples)*100:.1f}%)")

print("\n" + "="*80)
print("TỔNG HỢP THEO TRỤC Ý NGHĨA NHÃN (AXIS B):")
sem_counts = defaultdict(int)
for s in audited_samples:
    sem_counts[s['semantic']] += 1
for k, v in sem_counts.items():
    print(f"  {k:25s}: {v:3d} mẫu ({v/len(audited_samples)*100:.1f}%)")

print("\n" + "="*80)
print("TỔNG HỢP THEO TRẠNG THÁI ĐỦ ĐIỀU KIỆN (ELIGIBILITY):")
elig_counts = defaultdict(int)
for s in audited_samples:
    elig_counts[s['eligibility']] += 1
for k, v in elig_counts.items():
    print(f"  {k:25s}: {v:3d} mẫu ({v/len(audited_samples)*100:.1f}%)")

# Detailed per-class breakdown for 22 static classes
print("\n" + "="*80)
print("BẢNG CHI TIẾT TỪNG LỚP TRONG 22 CHỮ CÁI TĨNH:")
header = f"{'Lớp':3s} | {'Tổng':4s} | {'Tĩnh':4s} | {'BASE':4s} | {'Shared':6s} | {'South':5s} | {'North':5s} | {'E_Shared':8s} | {'E_South':7s} | {'Pending':7s} | {'Excluded':8s} | {'Rep_F':5s}"
print(header)
print("-" * len(header))

class_summary = {}
for c in classes:
    c_samples = [s for s in audited_samples if s['label'] == c]
    tot = len(c_samples)
    static_cnt = sum(1 for s in c_samples if s['source_type'] == 'static_video')
    base_cnt = sum(1 for s in c_samples if s['source_type'] == 'base_pose_of_dynamic')
    shared_cnt = sum(1 for s in c_samples if s['region'] == 'REGION_SHARED')
    south_cnt = sum(1 for s in c_samples if s['region'] == 'REGION_SOUTH')
    north_cnt = sum(1 for s in c_samples if s['region'] == 'REGION_NORTH')
    
    e_shared = sum(1 for s in c_samples if s['eligibility'] == 'ELIGIBLE_SHARED')
    e_south = sum(1 for s in c_samples if s['eligibility'] == 'ELIGIBLE_SOUTH')
    pending = sum(1 for s in c_samples if s['eligibility'] == 'PENDING_VERIFICATION')
    excluded = sum(1 for s in c_samples if s['eligibility'] in ('EXCLUDED_REGION', 'EXCLUDED_SEMANTIC'))
    rep_f = sum(1 for s in c_samples if not s['is_representative'])
    
    class_summary[c] = {
        'total': tot,
        'static': static_cnt,
        'base': base_cnt,
        'shared': shared_cnt,
        'south': south_cnt,
        'north': north_cnt,
        'e_shared': e_shared,
        'e_south': e_south,
        'pending': pending,
        'excluded': excluded,
        'rep_f': rep_f,
        'videos': list(set(s['video_id'] for s in c_samples))
    }
    
    print(f"{c:3s} | {tot:4d} | {static_cnt:4d} | {base_cnt:4d} | {shared_cnt:6d} | {south_cnt:5d} | {north_cnt:5d} | {e_shared:8d} | {e_south:7d} | {pending:7d} | {excluded:8d} | {rep_f:5d}")

# Save audited details to json for reporting
with open('scratch/task7_audited_samples.json', 'w', encoding='utf-8') as fp:
    json.dump({'samples': audited_samples, 'summary': class_summary}, fp, ensure_ascii=False, indent=2)

print("\nĐã lưu kết quả chi tiết vào scratch/task7_audited_samples.json")

