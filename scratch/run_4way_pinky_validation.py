#!/usr/bin/env python3
"""
Controlled 4-Way Validation: MediaPipe Pinky & Full-Hand Tracking
Configs:
- A  (Baseline):  model_complexity=0 (Lite), 320x240
- A2 (Res-Only):  model_complexity=0 (Lite), 480x360
- B  (Candidate): model_complexity=1 (Full), 480x360
- C  (Control):   model_complexity=1 (Full), 640x480

Evaluates on identical video sequences:
1. Folded_Pinky_D0532 (backend/data/media_cache/videos/D0532.mp4) - Shape W/3 (index, middle, ring extended; pinky folded)
2. Occluded_Pinky_Fist_D0032 (backend/data/media_cache/videos/D0032.mp4) - Closed fist (all 4 fingers folded/occluded)
3. Visible_Pinky_D0501 (backend/data/media_cache/videos/D0501.mp4) - Shape I (only pinky extended; index, middle, ring folded)
"""

import sys
import os
import time
import math
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
import mediapipe as mp
from backend.vision.feature_extractor import (
    _shift_and_scale,
    calculate_distance,
    calculate_angle,
    compute_finger_states,
    finger_curl_ratio,
)
from backend.vision.alphabet_classifier import alphabet_classifier


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
            return [[p['x'], p['y'], p['z']] for p in self.prev_x]

        dt = min(0.15, max(1.0 / 120.0, (ts_ms - self.prev_ts) / 1000.0))
        self.prev_ts = ts_ms
        a_d = smoothing_alpha(dt, self.d_cutoff)

        out = []
        new_dx = []
        for i in range(len(landmarks)):
            lm = landmarks[i]
            rx, ry, rz = float(lm[0]), float(lm[1]), float(lm[2])
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
        return [[p['x'], p['y'], p['z']] for p in out]


def compute_full_hand_metrics(landmarks):
    shifted, palm_scale = _shift_and_scale(landmarks)
    states = compute_finger_states(shifted, palm_scale)

    p0 = shifted[0]
    p17, p18, p19, p20 = shifted[17], shifted[18], shifted[19], shifted[20]

    # Pinky angles
    pip_ang = calculate_angle(p17, p18, p19)
    dip_ang = calculate_angle(p18, p19, p20)
    mcp_ang = calculate_angle(p0, p17, p18)

    # Other fingers PIP angles
    idx_pip_ang = calculate_angle(shifted[5], shifted[6], shifted[7])
    mid_pip_ang = calculate_angle(shifted[9], shifted[10], shifted[11])
    rng_pip_ang = calculate_angle(shifted[13], shifted[14], shifted[15])

    # Normalized tip-to-wrist distances
    d_tw = calculate_distance(shifted[4], p0) / palm_scale
    d_iw = calculate_distance(shifted[8], p0) / palm_scale
    d_mw = calculate_distance(shifted[12], p0) / palm_scale
    d_rw = calculate_distance(shifted[16], p0) / palm_scale
    d_pw = calculate_distance(p20, p0) / palm_scale

    # 3-segment pinky curl
    d_20_17 = calculate_distance(p20, p17)
    str_3seg = (
        calculate_distance(p20, p19)
        + calculate_distance(p19, p18)
        + calculate_distance(p18, p17)
    )
    curl_3seg = 1.0 - (d_20_17 / str_3seg) if str_3seg > 1e-6 else 0.0

    # Classifier evaluation
    cls_res = alphabet_classifier.classify_detailed(landmarks)

    return {
        'pip_angle': pip_ang,
        'dip_angle': dip_ang,
        'mcp_angle': mcp_ang,
        'idx_pip_angle': idx_pip_ang,
        'mid_pip_angle': mid_pip_ang,
        'rng_pip_angle': rng_pip_ang,
        'd_tw': d_tw,
        'd_iw': d_iw,
        'd_mw': d_mw,
        'd_rw': d_rw,
        'd_pw': d_pw,
        'thumb_curl': states['thumb_curl'],
        'index_curl': states['index_curl'],
        'middle_curl': states['middle_curl'],
        'ring_curl': states['ring_curl'],
        'pinky_curl_2seg': states['pinky_curl'],
        'pinky_curl_3seg': curl_3seg,
        'thumb_ext': states['thumb_extended'],
        'index_ext': states['index_extended'],
        'middle_ext': states['middle_extended'],
        'ring_ext': states['ring_extended'],
        'pinky_ext': states['pinky_extended'],
        'classified_letter': cls_res.get('letter'),
        'classified_status': cls_res.get('status'),
        'landmarks': landmarks,
    }


def run_config_on_video(video_path, model_complexity, input_res, max_frames=80):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    w_in, h_in = input_res
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.35,
    )
    one_euro = OneEuroPointSet(min_cutoff=2.2, beta=0.35, d_cutoff=1.0)

    latencies = []
    raw_frames = {}
    smooth_frames = {}
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame_bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        res = hands.process(rgb)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        ts_ms = frame_idx * 33.333
        if res.multi_hand_landmarks:
            lms_raw = [[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark]
            lms_smooth = one_euro.filter(lms_raw, ts_ms)
            raw_frames[frame_idx] = compute_full_hand_metrics(lms_raw)
            smooth_frames[frame_idx] = compute_full_hand_metrics(lms_smooth)
        else:
            one_euro.reset()

        frame_idx += 1

    cap.release()
    hands.close()

    # Compute frame-to-frame jitter on consecutive detected frames (non-pinky vs pinky)
    other_jitter_raw = []
    pinky_jitter_raw = []
    other_jitter_smooth = []
    pinky_jitter_smooth = []

    for f in sorted(raw_frames.keys()):
        if (f - 1) in raw_frames:
            curr_r = np.array(raw_frames[f]['landmarks'])
            prev_r = np.array(raw_frames[f - 1]['landmarks'])
            diffs_r = np.linalg.norm(curr_r[:, :2] - prev_r[:, :2], axis=1)
            other_jitter_raw.append(float(np.mean(diffs_r[:17])))
            pinky_jitter_raw.append(float(np.mean(diffs_r[17:21])))

            curr_s = np.array(smooth_frames[f]['landmarks'])
            prev_s = np.array(smooth_frames[f - 1]['landmarks'])
            diffs_s = np.linalg.norm(curr_s[:, :2] - prev_s[:, :2], axis=1)
            other_jitter_smooth.append(float(np.mean(diffs_s[:17])))
            pinky_jitter_smooth.append(float(np.mean(diffs_s[17:21])))

    return {
        'total_frames': frame_idx,
        'detected_frames': len(raw_frames),
        'detection_rate_pct': (len(raw_frames) / frame_idx * 100.0) if frame_idx else 0.0,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'python_cpu_fps': float(1000.0 / np.mean(latencies)) if latencies else 0.0,
        'raw_frames': raw_frames,
        'smooth_frames': smooth_frames,
        'other_jitter_raw': float(np.mean(other_jitter_raw)) if other_jitter_raw else 0.0,
        'pinky_jitter_raw': float(np.mean(pinky_jitter_raw)) if pinky_jitter_raw else 0.0,
        'other_jitter_smooth': float(np.mean(other_jitter_smooth)) if other_jitter_smooth else 0.0,
        'pinky_jitter_smooth': float(np.mean(pinky_jitter_smooth)) if pinky_jitter_smooth else 0.0,
    }


def summarize_video(video_name, video_path, expected_pinky_ext):
    configs = [
        ('A_Lite_320x240', 0, (320, 240)),
        ('A2_Lite_480x360', 0, (480, 360)),
        ('B_Full_480x360', 1, (480, 360)),
        ('C_Full_640x480', 1, (640, 480)),
    ]

    runs = {}
    for cfg_name, mc, res in configs:
        runs[cfg_name] = run_config_on_video(video_path, mc, res, max_frames=80)

    # Co-detected frames across all 4 configs
    common_f = set.intersection(*(set(runs[c[0]]['raw_frames'].keys()) for c in configs))
    common_f = sorted(common_f)

    summary = {
        'video_name': video_name,
        'common_detected_frames_count': len(common_f),
        'configs': {},
    }

    for cfg_name, mc, res in configs:
        r = runs[cfg_name]
        # Metrics over all detected frames for that config, plus over common frames
        rf_all = list(r['raw_frames'].values())
        sf_all = list(r['smooth_frames'].values())
        rf_com = [r['raw_frames'][f] for f in common_f]
        sf_com = [r['smooth_frames'][f] for f in common_f]

        def agg(items, key):
            vals = [float(x[key]) for x in items]
            return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'min': float(np.min(vals)), 'max': float(np.max(vals))}

        # Misprediction rate of pinky_ext vs expected_pinky_ext
        wrong_ext_all = [1.0 if (bool(x['pinky_ext']) != expected_pinky_ext) else 0.0 for x in sf_all]
        wrong_ext_com = [1.0 if (bool(x['pinky_ext']) != expected_pinky_ext) else 0.0 for x in sf_com]

        summary['configs'][cfg_name] = {
            'model_complexity': mc,
            'resolution': f"{res[0]}x{res[1]}",
            'detection_rate_pct': r['detection_rate_pct'],
            'detected_count': r['detected_frames'],
            'total_frames': r['total_frames'],
            'latency_mean_ms': r['latency_mean_ms'],
            'latency_std_ms': r['latency_std_ms'],
            'latency_p95_ms': r['latency_p95_ms'],
            'python_cpu_fps': r['python_cpu_fps'],
            'pinky_misprediction_pct_all': float(np.mean(wrong_ext_all) * 100.0) if wrong_ext_all else 0.0,
            'pinky_misprediction_pct_common': float(np.mean(wrong_ext_com) * 100.0) if wrong_ext_com else 0.0,
            'raw_common': {
                'pip_angle': agg(rf_com, 'pip_angle'),
                'dip_angle': agg(rf_com, 'dip_angle'),
                'mcp_angle': agg(rf_com, 'mcp_angle'),
                'd_pw': agg(rf_com, 'd_pw'),
                'pinky_curl_2seg': agg(rf_com, 'pinky_curl_2seg'),
                'pinky_curl_3seg': agg(rf_com, 'pinky_curl_3seg'),
            },
            'smooth_common': {
                'pip_angle': agg(sf_com, 'pip_angle'),
                'dip_angle': agg(sf_com, 'dip_angle'),
                'd_pw': agg(sf_com, 'd_pw'),
                'pinky_curl_2seg': agg(sf_com, 'pinky_curl_2seg'),
                'pinky_curl_3seg': agg(sf_com, 'pinky_curl_3seg'),
                'd_iw': agg(sf_com, 'd_iw'),
                'd_mw': agg(sf_com, 'd_mw'),
                'd_rw': agg(sf_com, 'd_rw'),
                'd_tw': agg(sf_com, 'd_tw'),
                'index_curl': agg(sf_com, 'index_curl'),
                'middle_curl': agg(sf_com, 'middle_curl'),
                'ring_curl': agg(sf_com, 'ring_curl'),
                'index_ext_pct': float(np.mean([float(x['index_ext']) for x in sf_com]) * 100.0),
                'middle_ext_pct': float(np.mean([float(x['middle_ext']) for x in sf_com]) * 100.0),
                'ring_ext_pct': float(np.mean([float(x['ring_ext']) for x in sf_com]) * 100.0),
                'pinky_ext_pct': float(np.mean([float(x['pinky_ext']) for x in sf_com]) * 100.0),
            },
            'stability': {
                'other_jitter_raw': r['other_jitter_raw'],
                'pinky_jitter_raw': r['pinky_jitter_raw'],
                'other_jitter_smooth': r['other_jitter_smooth'],
                'pinky_jitter_smooth': r['pinky_jitter_smooth'],
            }
        }

    return summary


if __name__ == '__main__':
    videos = [
        ('Folded_Pinky_D0532', 'backend/data/media_cache/videos/D0532.mp4', False),
        ('Occluded_Pinky_Fist_D0032', 'backend/data/media_cache/videos/D0032.mp4', False),
        ('Visible_Pinky_D0501', 'backend/data/media_cache/videos/D0501.mp4', True),
    ]

    results = []
    for name, path, exp_ext in videos:
        print(f"Processing {name}...")
        s = summarize_video(name, path, exp_ext)
        results.append(s)

    os.makedirs('scratch/ab_results', exist_ok=True)
    out_file = 'scratch/ab_results/four_way_validation.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print formatted tables
    for s in results:
        print(f"\n{'='*95}")
        print(f"VIDEO: {s['video_name']} (Co-detected frames N={s['common_detected_frames_count']})")
        print(f"{'='*95}")
        header = f"{'Config':<18} | {'Det%':>5} | {'Lat(ms)':>7} | {'P95(ms)':>7} | {'PyFPS':>6} | {'PIP°':>6} | {'DIP°':>6} | {'d_PW':>6} | {'Curl2':>6} | {'ErrExt%':>7}"
        print(header)
        print("-" * len(header))
        for cfg_name, c in s['configs'].items():
            sc = c['smooth_common']
            print(
                f"{cfg_name:<18} | {c['detection_rate_pct']:5.1f} | {c['latency_mean_ms']:7.2f} | "
                f"{c['latency_p95_ms']:7.2f} | {c['python_cpu_fps']:6.1f} | "
                f"{sc['pip_angle']['mean']:6.1f} | {sc['dip_angle']['mean']:6.1f} | "
                f"{sc['d_pw']['mean']:6.3f} | {sc['pinky_curl_2seg']['mean']:6.3f} | "
                f"{c['pinky_misprediction_pct_common']:6.1f}%"
            )
        print("\nOther Fingers Stability (Smoothed Common Frames):")
        header2 = f"{'Config':<18} | {'d_IW':>6} | {'d_MW':>6} | {'d_RW':>6} | {'IdxCurl':>7} | {'MidCurl':>7} | {'RngCurl':>7} | {'OthJit':>8} | {'PnkJit':>8}"
        print(header2)
        print("-" * len(header2))
        for cfg_name, c in s['configs'].items():
            sc = c['smooth_common']
            st = c['stability']
            print(
                f"{cfg_name:<18} | {sc['d_iw']['mean']:6.3f} | {sc['d_mw']['mean']:6.3f} | "
                f"{sc['d_rw']['mean']:6.3f} | {sc['index_curl']['mean']:7.3f} | "
                f"{sc['middle_curl']['mean']:7.3f} | {sc['ring_curl']['mean']:7.3f} | "
                f"{st['other_jitter_smooth']:8.5f} | {st['pinky_jitter_smooth']:8.5f}"
            )
