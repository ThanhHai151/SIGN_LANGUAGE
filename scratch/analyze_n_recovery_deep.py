import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.models.alphabet_static_head import load_static_alphabet_head, STATIC_CLASSES_22
from backend.vision.feature_extractor import normalize_hand_landmarks

head, _ = load_static_alphabet_head()

print("=== DEEP ANALYSIS OF N VS M / A / S / E ===")

# 1. Analyze Canonical N.json (D0505)
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_canonical = json.load(f)

n_samples = n_canonical.get('samples', [])
print(f"\n1. Canonical N (D0505): {len(n_samples)} samples")

n_confs = []
n_top1_classes = {}
n_top2_classes = {}
n_gaps = []
n_probs_record = []

for s in n_samples:
    lms = s['landmarks']
    feat_72, _ = normalize_hand_landmarks(lms)
    res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
    top_k = res['top_k']
    top1 = top_k[0]
    top2 = top_k[1]
    
    n_top1_classes[top1['letter']] = n_top1_classes.get(top1['letter'], 0) + 1
    n_top2_classes[top2['letter']] = n_top2_classes.get(top2['letter'], 0) + 1
    n_confs.append(top1['confidence'])
    gap = top1['confidence'] - top2['confidence']
    n_gaps.append(gap)
    
    # Get probs of N, M, A, S, E
    prob_dict = {item['letter']: item['confidence'] for item in top_k}
    n_probs_record.append({
        'frame': s.get('frame_index'),
        'top1': top1['letter'],
        'top1_conf': top1['confidence'],
        'top2': top2['letter'],
        'top2_conf': top2['confidence'],
        'gap': round(gap, 4),
        'P(N)': prob_dict.get('N', 0.0),
        'P(M)': prob_dict.get('M', 0.0),
        'P(A)': prob_dict.get('A', 0.0),
        'P(S)': prob_dict.get('S', 0.0),
        'P(E)': prob_dict.get('E', 0.0)
    })

print(f"Top-1 distribution in canonical N: {n_top1_classes}")
print(f"Top-2 distribution in canonical N: {n_top2_classes}")
print(f"Confidence: min={min(n_confs):.4f}, max={max(n_confs):.4f}, mean={np.mean(n_confs):.4f}, median={np.median(n_confs):.4f}")
print(f"Confidence Gap (Top1 - Top2): min={min(n_gaps):.4f}, max={max(n_gaps):.4f}, mean={np.mean(n_gaps):.4f}")
print(f"Frames with conf >= 0.70: {sum(1 for c in n_confs if c >= 0.70)} / {len(n_confs)} ({sum(1 for c in n_confs if c >= 0.70)/len(n_confs)*100:.1f}%)")
print(f"Frames with conf >= 0.60: {sum(1 for c in n_confs if c >= 0.60)} / {len(n_confs)} ({sum(1 for c in n_confs if c >= 0.60)/len(n_confs)*100:.1f}%)")
print(f"Frames with conf >= 0.50: {sum(1 for c in n_confs if c >= 0.50)} / {len(n_confs)} ({sum(1 for c in n_confs if c >= 0.50)/len(n_confs)*100:.1f}%)")

# 2. Analyze Canonical M.json (D0504)
with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_canonical = json.load(f)

m_samples = m_canonical.get('samples', [])
print(f"\n2. Canonical M (D0504): {len(m_samples)} samples")
m_top1_classes = {}
m_confs = []
m_n_conf = []
for s in m_samples:
    lms = s['landmarks']
    feat_72, _ = normalize_hand_landmarks(lms)
    res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
    top_k = res['top_k']
    top1 = top_k[0]
    m_top1_classes[top1['letter']] = m_top1_classes.get(top1['letter'], 0) + 1
    m_confs.append(top1['confidence'])
    p_n = next((item['confidence'] for item in top_k if item['letter'] == 'N'), 0.0)
    m_n_conf.append(p_n)

print(f"Top-1 distribution in canonical M: {m_top1_classes}")
print(f"M confidence: mean={np.mean(m_confs):.4f}, median={np.median(m_confs):.4f}")
print(f"P(N) when sample is M: max={max(m_n_conf):.4f}, mean={np.mean(m_n_conf):.4f}")

# 3. Analyze Canonical A, S, E
for other_letter in ['A', 'S', 'E']:
    with open(f'backend/data/alphabet_v2/static_landmarks/{other_letter}.json', 'r', encoding='utf-8') as f:
        other_data = json.load(f)
    samples = other_data.get('samples', [])
    p_n_list = []
    top1_list = {}
    for s in samples:
        lms = s['landmarks']
        feat_72, _ = normalize_hand_landmarks(lms)
        res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
        top_k = res['top_k']
        top1 = top_k[0]
        top1_list[top1['letter']] = top1_list.get(top1['letter'], 0) + 1
        p_n = next((item['confidence'] for item in top_k if item['letter'] == 'N'), 0.0)
        p_n_list.append(p_n)
    print(f"\n3. Canonical {other_letter}: {len(samples)} samples, Top1: {top1_list}, max P(N)={max(p_n_list):.4f}, mean P(N)={np.mean(p_n_list):.4f}")

# 4. Analyze Webcam N Log
webcam_n_lines = [json.loads(l) for l in open('backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl', encoding='utf-8')]
print(f"\n4. Webcam N Log: {len(webcam_n_lines)} frames")
webcam_confs = []
webcam_reasons = []
for f in webcam_n_lines:
    hc = f.get('head_confidence')
    if hc is not None:
        webcam_confs.append(float(hc))
    rej = f.get('reject_reason')
    if rej:
        webcam_reasons.append(rej)

print(f"Webcam N head_confidence (valid hand frames: {len(webcam_confs)}):")
print(f"  min={min(webcam_confs):.4f}, max={max(webcam_confs):.4f}, mean={np.mean(webcam_confs):.4f}, median={np.median(webcam_confs):.4f}")
print(f"  Frames >= 0.70: {sum(1 for c in webcam_confs if c >= 0.70)} ({sum(1 for c in webcam_confs if c >= 0.70)/len(webcam_confs)*100:.1f}%)")
print(f"  Frames >= 0.65: {sum(1 for c in webcam_confs if c >= 0.65)} ({sum(1 for c in webcam_confs if c >= 0.65)/len(webcam_confs)*100:.1f}%)")
print(f"  Frames >= 0.60: {sum(1 for c in webcam_confs if c >= 0.60)} ({sum(1 for c in webcam_confs if c >= 0.60)/len(webcam_confs)*100:.1f}%)")
print(f"  Frames >= 0.55: {sum(1 for c in webcam_confs if c >= 0.55)} ({sum(1 for c in webcam_confs if c >= 0.55)/len(webcam_confs)*100:.1f}%)")
print(f"  Frames >= 0.50: {sum(1 for c in webcam_confs if c >= 0.50)} ({sum(1 for c in webcam_confs if c >= 0.50)/len(webcam_confs)*100:.1f}%)")
print(f"  Frames >= 0.40: {sum(1 for c in webcam_confs if c >= 0.40)} ({sum(1 for c in webcam_confs if c >= 0.40)/len(webcam_confs)*100:.1f}%)")
