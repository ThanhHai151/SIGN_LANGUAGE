#!/usr/bin/env python3
"""
Controlled A/B Validation: MediaPipe Pinky Tracking
Config A: modelComplexity=0, input resolution 320x240 (Current baseline)
Config B: modelComplexity=1, input resolution 640x480 (Test hypothesis)

Runs on identical video frames from real QIPEDC videos:
- D0501.mp4: Ngón út duỗi thẳng (Visible Pinky - shape I)
- D0532.mp4: Ngón út gập sát lòng bàn tay (Folded Pinky - shape W/3)
- D0032.mp4: Nắm tay (Occluded Pinky in fist)
"""

import sys
import os
import time
import math
import json
import cv2
import numpy as np

# Ensure backend can be imported if needed
sys.path.insert(0, os.path.abspath('.'))

import mediapipe as mp

def smoothing_alpha(dt, cutoff):
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
    return 1.0 / (1.0 + tau / max(dt, 1e-4))

class OneEuroPointSet:
    def __init__(self, min_cutoff=2.2, beta=0.35, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.prev_x = None
        self.prev_dx = None
        self.prev_ts = None

    def reset(self):
        self.prev_x = None
        self.prev_dx = None
        self.prev_ts = None

    def filter(self, landmarks, ts_ms):
        if not landmarks:
            return None
        if self.prev_x is None or len(self.prev_x) != len(landmarks) or self.prev_ts is None:
            self.prev_x = [{'x': float(lm[0]), 'y': float(lm[1]), 'z': float(lm[2])} for lm in landmarks]
            self.prev_dx = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in landmarks]
            self.prev_ts = ts_ms
            return self.prev_x

        dt = min(0.15, max(1.0 / 120.0, (ts_ms - self.prev_ts) / 1000.0))
        self.prev_ts = ts_ms
        a_d = smoothing_alpha(dt, self.d_cutoff)

        out = []
        new_dx = []
        for i in range(len(landmarks)):
            lm = landmarks[i]
            rx = float(lm[0])
            ry = float(lm[1])
            rz = float(lm[2])
            px = self.prev_x[i]
            pd = self.prev_dx[i]

            dx_raw = (rx - px['x']) / dt
            dy_raw = (ry - px['y']) / dt
            dz_raw = (rz - px['z']) / dt

            dx = a_d * dx_raw + (1.0 - a_d) * pd['x']
            dy = a_d * dy_raw + (1.0 - a_d) * pd['y']
            dz = a_d * dz_raw + (1.0 - a_d) * pd['z']

            speed = math.hypot(dx, dy)
            cutoff = self.min_cutoff + self.beta * speed
            a = smoothing_alpha(dt, cutoff)

            fx = a * rx + (1.0 - a) * px['x']
            fy = a * ry + (1.0 - a) * px['y']
            fz = a * rz + (1.0 - a) * px['z']

            new_dx.append({'x': dx, 'y': dy, 'z': dz})
            out.append({'x': fx, 'y': fy, 'z': fz})

        self.prev_dx = new_dx
        self.prev_x = out
        return out

def calculate_distance(p1, p2):
    if not p1 or not p2:
        return 0.0
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    dz = float(p1[2]) - float(p2[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def calculate_angle(a, b, c):
    if not a or not b or not c:
        return 0.0
    ba = np.array([float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])], dtype=float)
    bc = np.array([float(c[0]) - float(b[0]), float(c[1]) - float(b[1]), float(c[2]) - float(b[2])], dtype=float)
    n_ba = np.linalg.norm(ba)
    n_bc = np.linalg.norm(bc)
    if n_ba < 1e-7 or n_bc < 1e-7:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def shift_and_scale(landmarks):
    wrist = landmarks[0]
    shifted = [[float(p[0]) - float(wrist[0]), float(p[1]) - float(wrist[1]), float(p[2]) - float(wrist[2])] for p in landmarks]
    palm_scale = calculate_distance(shifted[0], shifted[9])
    if palm_scale < 1e-5:
        palm_scale = 1.0
    return shifted, palm_scale

def compute_metrics(landmarks):
    shifted, palm_scale = shift_and_scale(landmarks)
    p17 = shifted[17] # MCP
    p18 = shifted[18] # PIP
    p19 = shifted[19] # DIP
    p20 = shifted[20] # TIP
    p0 = shifted[0]   # Wrist

    # Angles
    pip_ang = calculate_angle(p17, p18, p19)
    dip_ang = calculate_angle(p18, p19, p20)
    mcp_ang = calculate_angle(p0, p17, p18)

    # Distances
    d_pw = calculate_distance(p20, p0) / palm_scale
    d_tip_mcp = calculate_distance(p20, p17)
    d_tip_pip = calculate_distance(p20, p18)
    d_pip_mcp = calculate_distance(p18, p17)
    d_tip_dip = calculate_distance(p20, p19)
    d_dip_pip = calculate_distance(p19, p18)

    # Curl ratios
    # 2-segment curl (used in feature_extractor.py: skipping 19)
    str_len_2seg = d_tip_pip + d_pip_mcp
    curl_2seg = 1.0 - (d_tip_mcp / str_len_2seg) if str_len_2seg > 1e-6 else 0.0

    # 3-segment true anatomical curl
    str_len_3seg = d_tip_dip + d_dip_pip + d_pip_mcp
    curl_3seg = 1.0 - (d_tip_mcp / str_len_3seg) if str_len_3seg > 1e-6 else 0.0

    # Extended flag (shifted[20].y < shifted[18].y)
    p_ext = shifted[20][1] < shifted[18][1]

    return {
        'pip_angle': pip_ang,
        'dip_angle': dip_ang,
        'mcp_angle': mcp_ang,
        'd_pw': d_pw,
        'curl_2seg': curl_2seg,
        'curl_3seg': curl_3seg,
        'pinky_ext': p_ext,
        'raw_17': landmarks[17],
        'raw_18': landmarks[18],
        'raw_19': landmarks[19],
        'raw_20': landmarks[20],
    }

def run_pipeline(video_path, model_complexity, input_res, max_frames=80):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w_in, h_in = input_res
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.35
    )
    filter_set = OneEuroPointSet(min_cutoff=2.2, beta=0.35, d_cutoff=1.0)

    frame_count = 0
    latencies = []
    raw_results = []
    smooth_results = []
    frames_rgb = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize to test input resolution
        resized_bgr = cv2.resize(frame_bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        res = hands.process(rgb)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms

        ts_ms = frame_count * 33.333
        if res.multi_hand_landmarks:
            # Pick primary hand (first hand)
            lms_raw = [[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark]
            raw_metrics = compute_metrics(lms_raw)

            # Apply One-Euro filter
            lms_smooth_dict = filter_set.filter(lms_raw, ts_ms)
            lms_smooth = [[lm['x'], lm['y'], lm['z']] for lm in lms_smooth_dict]
            smooth_metrics = compute_metrics(lms_smooth)

            raw_results.append((frame_count, raw_metrics))
            smooth_results.append((frame_count, smooth_metrics))
        else:
            filter_set.reset()
            raw_results.append((frame_count, None))
            smooth_results.append((frame_count, None))

        frames_rgb.append(frame_bgr)
        frame_count += 1

    cap.release()
    hands.close()

    return {
        'latencies_ms': latencies,
        'raw_results': raw_results,
        'smooth_results': smooth_results,
        'frames': frames_rgb
    }

def draw_skeleton_overlay(img_bgr, landmarks, color=(0, 255, 0)):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    HAND_BONES = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    # Draw bones
    for i, j in HAND_BONES:
        p1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        p2 = (int(landmarks[j][0] * w), int(landmarks[j][1] * h))
        # Highlight pinky bones in bright magenta/yellow
        bone_color = (0, 0, 255) if (i >= 17 or j >= 17) else color
        thickness = 3 if (i >= 17 or j >= 17) else 2
        cv2.line(out, p1, p2, bone_color, thickness)

    # Draw landmarks
    for idx, p in enumerate(landmarks):
        pt = (int(p[0] * w), int(p[1] * h))
        pt_color = (0, 255, 255) if idx in (17, 18, 19, 20) else (255, 255, 255)
        radius = 4 if idx in (17, 18, 19, 20) else 3
        cv2.circle(out, pt, radius, pt_color, -1)

    return out

def run_ab_comparison(video_name, video_path):
    print(f"\n{'='*75}")
    print(f"RUNNING CONTROLLED A/B TEST ON: {video_name} ({video_path})")
    print(f"{'='*75}")

    print("Running Config A: modelComplexity=0, res=320x240...")
    res_a = run_pipeline(video_path, model_complexity=0, input_res=(320, 240))

    print("Running Config B: modelComplexity=1, res=640x480...")
    res_b = run_pipeline(video_path, model_complexity=1, input_res=(640, 480))

    # Latency comparison
    lat_a = res_a['latencies_ms']
    lat_b = res_b['latencies_ms']
    fps_a = 1000.0 / np.mean(lat_a) if lat_a else 0
    fps_b = 1000.0 / np.mean(lat_b) if lat_b else 0

    print("\n--- 1. PERFORMANCE & LATENCY ---")
    print(f"Config A (Lite 320x240): Mean Latency = {np.mean(lat_a):.2f} ms (±{np.std(lat_a):.2f} ms) -> ~{fps_a:.1f} FPS")
    print(f"Config B (Full 640x480): Mean Latency = {np.mean(lat_b):.2f} ms (±{np.std(lat_b):.2f} ms) -> ~{fps_b:.1f} FPS")
    print(f"Latency delta: +{np.mean(lat_b) - np.mean(lat_a):.2f} ms ({np.mean(lat_b)/np.mean(lat_a):.2f}x)")

    # Analyze Pinky Metrics on Detected Frames
    valid_frames = []
    pip_a, pip_b = [], []
    dip_a, dip_b = [], []
    mcp_a, mcp_b = [], []
    dpw_a, dpw_b = [], []
    curl2_a, curl2_b = [], []
    curl3_a, curl3_b = [], []
    ext_a, ext_b = [], []

    raw_a_dict = dict(res_a['raw_results'])
    raw_b_dict = dict(res_b['raw_results'])

    for f_idx in range(len(res_a['raw_results'])):
        ma = raw_a_dict.get(f_idx)
        mb = raw_b_dict.get(f_idx)
        if ma and mb:
            valid_frames.append(f_idx)
            pip_a.append(ma['pip_angle'])
            pip_b.append(mb['pip_angle'])
            dip_a.append(ma['dip_angle'])
            dip_b.append(mb['dip_angle'])
            mcp_a.append(ma['mcp_angle'])
            mcp_b.append(mb['mcp_angle'])
            dpw_a.append(ma['d_pw'])
            dpw_b.append(mb['d_pw'])
            curl2_a.append(ma['curl_2seg'])
            curl2_b.append(mb['curl_2seg'])
            curl3_a.append(ma['curl_3seg'])
            curl3_b.append(mb['curl_3seg'])
            ext_a.append(int(ma['pinky_ext']))
            ext_b.append(int(mb['pinky_ext']))

    det_rate_a = sum(1 for _, m in res_a['raw_results'] if m) / len(res_a['raw_results']) * 100
    det_rate_b = sum(1 for _, m in res_b['raw_results'] if m) / len(res_b['raw_results']) * 100

    print("\n--- 2. DETECTION RATE ---")
    print(f"Config A Detection Rate: {det_rate_a:.1f}% ({sum(1 for _, m in res_a['raw_results'] if m)}/{len(res_a['raw_results'])})")
    print(f"Config B Detection Rate: {det_rate_b:.1f}% ({sum(1 for _, m in res_b['raw_results'] if m)}/{len(res_b['raw_results'])})")

    if valid_frames:
        print("\n--- 3. PINKY METRICS COMPARISON (CO-DETECTED FRAMES, N={len(valid_frames)}) ---")
        print(f"{'Metric':<22} | {'Config A (Lite 320x240)':<25} | {'Config B (Full 640x480)':<25} | {'Delta (B - A)'}")
        print("-" * 85)
        print(f"{'PIP Angle (deg)':<22} | mean={np.mean(pip_a):6.1f}° (std={np.std(pip_a):4.1f}°) | mean={np.mean(pip_b):6.1f}° (std={np.std(pip_b):4.1f}°) | {np.mean(pip_b)-np.mean(pip_a):+6.1f}°")
        print(f"{'DIP Angle (deg)':<22} | mean={np.mean(dip_a):6.1f}° (std={np.std(dip_a):4.1f}°) | mean={np.mean(dip_b):6.1f}° (std={np.std(dip_b):4.1f}°) | {np.mean(dip_b)-np.mean(dip_a):+6.1f}°")
        print(f"{'MCP Angle (deg)':<22} | mean={np.mean(mcp_a):6.1f}° (std={np.std(mcp_a):4.1f}°) | mean={np.mean(mcp_b):6.1f}° (std={np.std(mcp_b):4.1f}°) | {np.mean(mcp_b)-np.mean(mcp_a):+6.1f}°")
        print(f"{'d_PW / palm_scale':<22} | mean={np.mean(dpw_a):6.3f}  (std={np.std(dpw_a):4.3f})  | mean={np.mean(dpw_b):6.3f}  (std={np.std(dpw_b):4.3f})  | {np.mean(dpw_b)-np.mean(dpw_a):+6.3f}")
        print(f"{'Curl 2-seg':<22} | mean={np.mean(curl2_a):6.3f}  (std={np.std(curl2_a):4.3f})  | mean={np.mean(curl2_b):6.3f}  (std={np.std(curl2_b):4.3f})  | {np.mean(curl2_b)-np.mean(curl2_a):+6.3f}")
        print(f"{'Curl 3-seg':<22} | mean={np.mean(curl3_a):6.3f}  (std={np.std(curl3_a):4.3f})  | mean={np.mean(curl3_b):6.3f}  (std={np.std(curl3_b):4.3f})  | {np.mean(curl3_b)-np.mean(curl3_a):+6.3f}")
        print(f"{'Pinky Extended (%)':<22} | {np.mean(ext_a)*100:6.1f}%                   | {np.mean(ext_b)*100:6.1f}%                   | {(np.mean(ext_b)-np.mean(ext_a))*100:+6.1f}%")

    # Export a few sample visual comparison crops
    out_dir = os.path.join('scratch', 'ab_results')
    os.makedirs(out_dir, exist_ok=True)

    # Pick representative frame (middle frame with stable hand)
    if valid_frames:
        rep_idx = valid_frames[len(valid_frames) // 2]
        orig_frame = res_a['frames'][rep_idx]
        h, w = orig_frame.shape[:2]

        lms_a = [m['raw_17'] for f, m in res_a['raw_results'] if f == rep_idx]
        # Get full 21 landmarks for rep_idx
        # Redraw
        hands_a = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0, min_detection_confidence=0.3)
        hands_b = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1, min_detection_confidence=0.3)

        ra = hands_a.process(cv2.cvtColor(cv2.resize(orig_frame, (320, 240)), cv2.COLOR_BGR2RGB))
        rb = hands_b.process(cv2.cvtColor(cv2.resize(orig_frame, (640, 480)), cv2.COLOR_BGR2RGB))

        vis_a = orig_frame.copy()
        vis_b = orig_frame.copy()

        if ra.multi_hand_landmarks:
            la = [[lm.x, lm.y, lm.z] for lm in ra.multi_hand_landmarks[0].landmark]
            vis_a = draw_skeleton_overlay(vis_a, la, color=(0, 255, 0))
            cv2.putText(vis_a, f"Config A (Lite 320x240) - Frame {rep_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if rb.multi_hand_landmarks:
            lb = [[lm.x, lm.y, lm.z] for lm in rb.multi_hand_landmarks[0].landmark]
            vis_b = draw_skeleton_overlay(vis_b, lb, color=(255, 200, 0))
            cv2.putText(vis_b, f"Config B (Full 640x480) - Frame {rep_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        hands_a.close()
        hands_b.close()

        side_by_side = np.hstack([vis_a, vis_b])
        out_path = os.path.join(out_dir, f"{video_name}_frame_{rep_idx:03d}_comparison.jpg")
        cv2.imwrite(out_path, side_by_side)
        print(f"\nSaved visual side-by-side comparison to: {out_path}")

    return {
        'video_name': video_name,
        'fps_a': fps_a,
        'fps_b': fps_b,
        'lat_a_mean': float(np.mean(lat_a)),
        'lat_b_mean': float(np.mean(lat_b)),
        'det_rate_a': det_rate_a,
        'det_rate_b': det_rate_b,
        'metrics_count': len(valid_frames),
        'pip_a_mean': float(np.mean(pip_a)) if valid_frames else 0.0,
        'pip_b_mean': float(np.mean(pip_b)) if valid_frames else 0.0,
        'dip_a_mean': float(np.mean(dip_a)) if valid_frames else 0.0,
        'dip_b_mean': float(np.mean(dip_b)) if valid_frames else 0.0,
        'dpw_a_mean': float(np.mean(dpw_a)) if valid_frames else 0.0,
        'dpw_b_mean': float(np.mean(dpw_b)) if valid_frames else 0.0,
        'curl2_a_mean': float(np.mean(curl2_a)) if valid_frames else 0.0,
        'curl2_b_mean': float(np.mean(curl2_b)) if valid_frames else 0.0,
    }

if __name__ == '__main__':
    videos = [
        ('Folded_Pinky_D0532', 'backend/data/media_cache/videos/D0532.mp4'),
        ('Occluded_Pinky_Fist_D0032', 'backend/data/media_cache/videos/D0032.mp4'),
        ('Visible_Pinky_D0501', 'backend/data/media_cache/videos/D0501.mp4')
    ]

    all_results = []
    for name, path in videos:
        if os.path.exists(path):
            res = run_ab_comparison(name, path)
            all_results.append(res)
        else:
            print(f"Skipping {name}, file not found: {path}")

    report_path = os.path.join('scratch', 'ab_results', 'ab_experiment_summary.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCompleted A/B experiment. Summary written to {report_path}")
