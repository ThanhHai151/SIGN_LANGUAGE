"""
VSL-P010 — Script 3: Pose & Temporal Audit
Phân tích định lượng:
- Thống kê vai/khuỷu tay cố định trong tensor hands_fallback_pose
- So sánh window_duration_ms giữa holistic_raw và hands_fallback_pose
- Đặt câu hỏi về FPS/sampling/normalization

Read-only. Không sửa code runtime.
"""
import os, sys, json, glob, csv
from collections import defaultdict
import numpy as np

PROJECT_ROOT = r'E:\NAM4\SEMINAR\SIGN_LANGUAGE'
sys.path.insert(0, PROJECT_ROOT)

DEBUG_DIR  = os.path.join(PROJECT_ROOT, 'debug', 'words')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'vsl', 'word_ab_fixed_input')
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSE_END = 33
LH_START, LH_END = 33, 54
RH_START, RH_END = 54, 75
SHOULDER_L, SHOULDER_R = 11, 12
ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16

# Hardcoded fallback coordinates from CameraSignRecognizer.jsx (DEFAULT_POSE)
DEFAULT_SHOULDER_L = np.array([0.65, 0.45, 0.0])
DEFAULT_SHOULDER_R = np.array([0.35, 0.45, 0.0])


def analyze_tensor_pose(arr: np.ndarray) -> dict:
    """
    Phân tích tensor (30, 75, 3).
    Trả về dict thống kê về tính cố định của pose landmarks.
    """
    pose = arr[:, :POSE_END, :]          # (30, 33, 3)
    lh   = arr[:, LH_START:LH_END, :]   # (30, 21, 3)
    rh   = arr[:, RH_START:RH_END, :]   # (30, 21, 3)
    
    # Độ lệch chuẩn theo thời gian cho từng joint (30 frames)
    pose_std_per_joint = pose.std(axis=0)  # (33, 3)
    pose_max_std = pose_std_per_joint.max(axis=-1)  # (33,) - max std across xyz
    
    # Vai
    sh_l_std = float(pose[:, SHOULDER_L, :].std())
    sh_r_std = float(pose[:, SHOULDER_R, :].std())
    el_l_std = float(pose[:, ELBOW_L, :].std())
    el_r_std = float(pose[:, ELBOW_R, :].std())
    
    # Kiểm tra gần DEFAULT_POSE fallback
    sh_l_first = pose[0, SHOULDER_L, :]
    sh_r_first = pose[0, SHOULDER_R, :]
    dist_sh_l_from_default = float(np.linalg.norm(sh_l_first - DEFAULT_SHOULDER_L))
    dist_sh_r_from_default = float(np.linalg.norm(sh_r_first - DEFAULT_SHOULDER_R))
    is_default_pose = dist_sh_l_from_default < 0.02 and dist_sh_r_from_default < 0.02
    
    # Số joint pose cố định (std < threshold)
    STATIC_THR = 1e-4
    n_static_joints = int((pose_max_std < STATIC_THR).sum())
    
    # Missing landmark analysis
    lh_zero_frames = int((lh.reshape(30, -1).sum(axis=-1) == 0).sum())
    rh_zero_frames = int((rh.reshape(30, -1).sum(axis=-1) == 0).sum())
    both_zero = int(((lh.reshape(30, -1).sum(axis=-1) == 0) & (rh.reshape(30, -1).sum(axis=-1) == 0)).sum())
    
    # Total missing landmarks ratio
    total_vals = arr.size
    n_nonzero = int(np.any(arr != 0, axis=-1).sum())  # landmark có ít nhất 1 non-zero coord
    # Landmark-level zero: (30, 75) matrix
    landmark_zero = (arr == 0).all(axis=-1)  # (30, 75) bool
    total_missing_ratio = float(landmark_zero.mean())
    
    return {
        'sh_l_std': round(sh_l_std, 6),
        'sh_r_std': round(sh_r_std, 6),
        'el_l_std': round(el_l_std, 6),
        'el_r_std': round(el_r_std, 6),
        'n_static_pose_joints': n_static_joints,
        'is_near_default_fallback': is_default_pose,
        'dist_sh_l_from_default': round(dist_sh_l_from_default, 6),
        'dist_sh_r_from_default': round(dist_sh_r_from_default, 6),
        'lh_zero_frames': lh_zero_frames,
        'rh_zero_frames': rh_zero_frames,
        'both_hands_zero_frames': both_zero,
        'total_missing_ratio': round(total_missing_ratio, 4),
    }


def collect_records_with_pose():
    rows = []
    for session_json_path in sorted(glob.glob(os.path.join(DEBUG_DIR, '*', 'session.json'))):
        session_dir = os.path.dirname(session_json_path)
        with open(session_json_path, encoding='utf-8') as f:
            meta = json.load(f)
        
        events_path = os.path.join(session_dir, 'events.jsonl')
        if not os.path.exists(events_path):
            continue
        
        pred_events = []
        with open(events_path, encoding='utf-8') as f:
            for line in f:
                try:
                    ev = json.loads(line.strip())
                    if ev.get('event_type') == 'PREDICTION':
                        pred_events.append(ev)
                except:
                    pass
        
        for ev in pred_events:
            seq_file = ev.get('sequence_file', '')
            npy_path = os.path.join(session_dir, seq_file)
            if not os.path.exists(npy_path):
                continue
            arr = np.load(npy_path)
            if arr.shape != (30, 75, 3):
                continue
            
            pose_stats = analyze_tensor_pose(arr)
            rows.append({
                'session_id': meta.get('session_id', ''),
                'target_word': meta.get('target_word', ''),
                'ckpt_type': meta.get('checkpoint', {}).get('checkpoint_type', ''),
                'window_id': ev.get('window_id'),
                'landmarks_source': ev.get('landmarks_source', 'unknown'),
                'window_duration_ms': ev.get('window_duration_ms'),
                'client_fps': ev.get('client_fps'),
                'inference_latency_ms': ev.get('inference_latency_ms'),
                # pose stats từ log (sẵn có)
                'log_sh_l_std': ev.get('shoulder_left_std', 0),
                'log_sh_r_std': ev.get('shoulder_right_std', 0),
                'log_is_static': ev.get('is_static_shoulders', False),
                'log_fixed_pose_count': ev.get('fixed_pose_landmarks_count', 0),
                'log_total_missing': ev.get('total_missing_landmarks_ratio', 0),
                # pose stats recomputed từ NPY
                **{f'npy_{k}': v for k, v in pose_stats.items()},
            })
    return rows


if __name__ == '__main__':
    print("=== VSL-P010 POSE & TEMPORAL AUDIT ===")
    
    rows = collect_records_with_pose()
    print(f"Records collected: {len(rows)}")
    
    holistic = [r for r in rows if r['landmarks_source'] == 'holistic_raw']
    fallback  = [r for r in rows if r['landmarks_source'] == 'hands_fallback_pose']
    
    print(f"\n--- PHÂN NHÓM LANDMARK SOURCE ---")
    print(f"holistic_raw:        {len(holistic)} windows")
    print(f"hands_fallback_pose: {len(fallback)} windows")
    
    # Window duration
    h_dur = [r['window_duration_ms'] for r in holistic if r['window_duration_ms']]
    f_dur = [r['window_duration_ms'] for r in fallback if r['window_duration_ms']]
    h_fps = [r['client_fps'] for r in holistic if r['client_fps']]
    f_fps = [r['client_fps'] for r in fallback if r['client_fps']]
    
    print(f"\n--- WINDOW DURATION (ms) ---")
    if h_dur:
        print(f"holistic_raw:        mean={np.mean(h_dur):.0f} median={np.median(h_dur):.0f} "
              f"std={np.std(h_dur):.0f} min={min(h_dur)} max={max(h_dur)}")
    if f_dur:
        print(f"hands_fallback_pose: mean={np.mean(f_dur):.0f} median={np.median(f_dur):.0f} "
              f"std={np.std(f_dur):.0f} min={min(f_dur)} max={max(f_dur)}")
    
    print(f"\n--- CLIENT FPS ---")
    if h_fps:
        print(f"holistic_raw:        mean={np.mean(h_fps):.1f} median={np.median(h_fps):.1f} "
              f"std={np.std(h_fps):.1f} min={min(h_fps)} max={max(h_fps)}")
    if f_fps:
        print(f"hands_fallback_pose: mean={np.mean(f_fps):.1f} median={np.median(f_fps):.1f} "
              f"std={np.std(f_fps):.1f} min={min(f_fps)} max={max(f_fps)}")
    
    print(f"\n--- STATIC SHOULDER ANALYSIS (hands_fallback_pose, n={len(fallback)}) ---")
    if fallback:
        near_def = sum(1 for r in fallback if r['npy_is_near_default_fallback'])
        log_static = sum(1 for r in fallback if r['log_is_static'])
        avg_sh_l_std = np.mean([r['npy_sh_l_std'] for r in fallback])
        avg_sh_r_std = np.mean([r['npy_sh_r_std'] for r in fallback])
        avg_static_joints = np.mean([r['npy_n_static_pose_joints'] for r in fallback])
        print(f"  Near DEFAULT_POSE fallback coords: {near_def}/{len(fallback)} ({100*near_def/len(fallback):.1f}%)")
        print(f"  Log flagged is_static_shoulders:   {log_static}/{len(fallback)}")
        print(f"  Avg shoulder_left_std  (NPY): {avg_sh_l_std:.6f}")
        print(f"  Avg shoulder_right_std (NPY): {avg_sh_r_std:.6f}")
        print(f"  Avg static pose joints (std<1e-4): {avg_static_joints:.1f}/33")
    
    print(f"\n--- STATIC SHOULDER ANALYSIS (holistic_raw, n={len(holistic)}) ---")
    if holistic:
        avg_sh_l_std = np.mean([r['npy_sh_l_std'] for r in holistic])
        avg_sh_r_std = np.mean([r['npy_sh_r_std'] for r in holistic])
        avg_static_joints = np.mean([r['npy_n_static_pose_joints'] for r in holistic])
        print(f"  Avg shoulder_left_std  (NPY): {avg_sh_l_std:.6f}")
        print(f"  Avg shoulder_right_std (NPY): {avg_sh_r_std:.6f}")
        print(f"  Avg static pose joints (std<1e-4): {avg_static_joints:.1f}/33")
    
    # Lưu CSV
    if rows:
        fieldnames = list(rows[0].keys())
        csv_path = os.path.join(OUTPUT_DIR, '_pose_temporal_audit_raw.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nData saved: {csv_path}")
    
    print("\nAudit hoàn tất.")
    
    # Lưu summary dict cho report
    summary = {
        'n_holistic': len(holistic),
        'n_fallback': len(fallback),
        'holistic_duration_mean_ms': round(float(np.mean(h_dur)), 1) if h_dur else None,
        'holistic_duration_median_ms': round(float(np.median(h_dur)), 1) if h_dur else None,
        'fallback_duration_mean_ms': round(float(np.mean(f_dur)), 1) if f_dur else None,
        'fallback_duration_median_ms': round(float(np.median(f_dur)), 1) if f_dur else None,
        'holistic_fps_mean': round(float(np.mean(h_fps)), 1) if h_fps else None,
        'fallback_fps_mean': round(float(np.mean(f_fps)), 1) if f_fps else None,
        'fallback_near_default_pct': round(100*near_def/len(fallback), 1) if fallback else None,
        'fallback_avg_sh_l_std': round(float(np.mean([r['npy_sh_l_std'] for r in fallback])), 7) if fallback else None,
        'fallback_avg_sh_r_std': round(float(np.mean([r['npy_sh_r_std'] for r in fallback])), 7) if fallback else None,
        'fallback_avg_static_joints': round(float(np.mean([r['npy_n_static_pose_joints'] for r in fallback])), 1) if fallback else None,
        'holistic_avg_sh_l_std': round(float(np.mean([r['npy_sh_l_std'] for r in holistic])), 7) if holistic else None,
        'holistic_avg_static_joints': round(float(np.mean([r['npy_n_static_pose_joints'] for r in holistic])), 1) if holistic else None,
    }
    with open(os.path.join(OUTPUT_DIR, '_pose_audit_summary.json'), 'w', encoding='utf-8') as f:
        import json as _json
        _json.dump(summary, f, ensure_ascii=False, indent=2)

