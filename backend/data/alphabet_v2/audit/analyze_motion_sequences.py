import sys
import os
import json
import math

sys.stdout.reconfigure(encoding='utf-8')

motion_dir = 'backend/data/alphabet_v2/motion_sequences'
letters = ['Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ']

def dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

analysis_results = {}

for l in letters:
    fpath = os.path.join(motion_dir, f"{l}.json")
    with open(fpath, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    seq = data['sequence']
    total_frames = len(seq)
    
    # 1. Inspect valid/invalid segments
    status_list = [s['quality_status'] for s in seq]
    ok_indices = [s['frame_index'] for s in seq if s['quality_status'] == 'ok']
    
    # Group contiguous blocks of OK vs NO_HAND
    blocks = []
    if seq:
        cur_status = seq[0]['quality_status']
        start_idx = seq[0]['frame_index']
        prev_idx = start_idx
        for s in seq[1:]:
            fidx = s['frame_index']
            st = s['quality_status']
            if st == cur_status and fidx == prev_idx + 1:
                prev_idx = fidx
            else:
                blocks.append((cur_status, start_idx, prev_idx, prev_idx - start_idx + 1))
                cur_status = st
                start_idx = fidx
                prev_idx = fidx
        blocks.append((cur_status, start_idx, prev_idx, prev_idx - start_idx + 1))
    
    # 2. Kinematic analysis on valid frames
    # Track wrist (0), index_tip (8), thumb_tip (4), pinky_tip (20)
    velocities = [] # (fidx, wrist_disp, index_disp, pinky_disp)
    prev_s = None
    for s in seq:
        if s['quality_status'] == 'ok':
            if prev_s is not None and prev_s['quality_status'] == 'ok' and s['frame_index'] == prev_s['frame_index'] + 1:
                l1 = prev_s['landmarks']
                l2 = s['landmarks']
                # Palm scale normalization
                palm_scale = dist(l2[0], l2[9]) or 1.0
                w_v = dist(l1[0], l2[0]) / palm_scale
                i_v = dist(l1[8], l2[8]) / palm_scale
                p_v = dist(l1[20], l2[20]) / palm_scale
                t_v = dist(l1[4], l2[4]) / palm_scale
                velocities.append({
                    'frame': s['frame_index'],
                    'timestamp': s['timestamp_sec'],
                    'wrist_v': round(w_v, 4),
                    'index_v': round(i_v, 4),
                    'pinky_v': round(p_v, 4),
                    'thumb_v': round(t_v, 4),
                    'wrist_y': round(l2[0][1], 4),
                    'wrist_z': round(l2[0][2], 4)
                })
            prev_s = s
        else:
            prev_s = s

    analysis_results[l] = {
        'total_frames': total_frames,
        'ok_count': len(ok_indices),
        'blocks': blocks,
        'velocities': velocities
    }

print("=== HOÀN TẤT TRÍCH XUẤT ĐẶC TRƯNG ĐỘNG HỌC 7 CHỮ ===")
for l in letters:
    res = analysis_results[l]
    print(f"\n--- CHỮ {l} (Tổng {res['total_frames']} frame, OK: {res['ok_count']}) ---")
    print("Các phân đoạn trạng thái (status, start_f, end_f, len):")
    for b in res['blocks']:
        print(f"  {b[0]:<18}: frame {b[1]:3d} -> {b[2]:3d} ({b[3]:2d} frames)")
    vels = res['velocities']
    if vels:
        avg_w = sum(v['wrist_v'] for v in vels) / len(vels)
        max_w = max(vels, key=lambda x: x['wrist_v'])
        max_i = max(vels, key=lambda x: x['index_v'])
        max_p = max(vels, key=lambda x: x['pinky_v'])
        print(f"  Vận tốc cổ tay TB: {avg_w:.4f}")
        print(f"  Đỉnh vận tốc cổ tay: frame {max_w['frame']} (v={max_w['wrist_v']:.4f})")
        print(f"  Đỉnh vận tốc ngón trỏ: frame {max_i['frame']} (v={max_i['index_v']:.4f})")
        print(f"  Đỉnh vận tốc ngón út: frame {max_p['frame']} (v={max_p['pinky_v']:.4f})")

with open('backend/data/alphabet_v2/audit/motion_kinematics.json', 'w', encoding='utf-8') as fp:
    json.dump(analysis_results, fp, ensure_ascii=False, indent=2)

print("\nĐã lưu kết quả phân tích động học vào: backend/data/alphabet_v2/audit/motion_kinematics.json")

