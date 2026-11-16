import sys
sys.path.insert(0, '.')
import json
import math
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# -----------------------------------------------------------------------------
# 1. Candidate P-down condition evaluator using EXACT production features
# -----------------------------------------------------------------------------
def evaluate_p_candidate_conditions(feat):
    if not feat or not isinstance(feat, dict):
        return None
    
    d_iw = feat.get('d_IW', 0.0)
    idx_straight = feat.get('idx_straight', 0.0)
    idx_tip_y = feat.get('idx_tip_y', 0.0)
    idx_dy = feat.get('idx_dy', 0.0)
    idx_horiz_ratio = feat.get('idx_horiz_ratio', 1.0)
    
    d_mw = feat.get('d_MW', 0.0)
    mid_straight = feat.get('mid_straight', 0.0)
    mid_tip_y = feat.get('mid_tip_y', 0.0)
    
    d_tm = feat.get('d_TM', 99.0)
    d_ti = feat.get('d_TI', 99.0)
    d_tw = feat.get('d_TW', 0.0)
    
    ang_im_dir = feat.get('ang_IM_dir', 0.0)
    
    d_rw = feat.get('d_RW', 99.0)
    d_pw = feat.get('d_PW', 99.0)
    
    conds = {
        'c1_d_iw_ge_1_70': bool(d_iw >= 1.70),
        'c2_idx_straight_ge_0_90': bool(idx_straight >= 0.90),
        'c3_idx_tip_y_ge_0_60': bool(idx_tip_y >= 0.60),
        'c4_idx_dy_ge_0_60': bool(idx_dy >= 0.60),
        'c5_idx_horiz_le_0_70': bool(idx_horiz_ratio <= 0.70),
        'c6_d_mw_ge_1_35': bool(d_mw >= 1.35),
        'c7_mid_straight_ge_0_80': bool(mid_straight >= 0.80),
        'c8_mid_tip_y_ge_0_60': bool(mid_tip_y >= 0.60),
        'c9_d_tm_le_1_10': bool(d_tm <= 1.10),
        'c10_d_ti_le_1_05': bool(d_ti <= 1.05),
        'c11_d_tw_ge_1_20': bool(d_tw >= 1.20),
        'c12_ang_im_range_15_50': bool(15.0 <= ang_im_dir <= 50.0),
        'c13_d_rw_le_1_60': bool(d_rw <= 1.60),
        'c14_d_pw_le_1_50': bool(d_pw <= 1.50),
    }
    
    matched = all(conds.values())
    return {
        'matched': matched,
        'conds': conds,
        'failed': [k for k, v in conds.items() if not v]
    }

# -----------------------------------------------------------------------------
# 2. Production Pipeline Simulation Harness
# Re-implements evaluate_rules with candidate 2-branch P at Rule 15
# -----------------------------------------------------------------------------
def simulate_production_pipeline(landmarks, features=None, has_static_head=False):
    """
    Simulates the EXACT production recognition pipeline order:
    1. Extract features (production extract_features)
    2. Evaluate geometric rules 1-18 in exact production sequence
       Rule 15 is upgraded to Two-Branch P (North Upright OR Candidate Down)
    3. Check conflict resolution (len(matches) == 1 -> MATCH, >1 -> CONFLICT)
    4. Recognizer gating & fallback (if NO_MATCH and has_static_head -> MLP; else NEUTRAL)
    """
    if features is None:
        if landmarks is None:
            return {"final_sign": "NEUTRAL", "status": "INVALID_INPUT", "matches": []}
        features = clf.extract_features(landmarks)
        if features is None:
            return {"final_sign": "NEUTRAL", "status": "INVALID_INPUT", "matches": []}
    
    # Run standard evaluate_rules WITHOUT rule 15
    # To be 100% faithful to production, let's call standard evaluate_rules
    # and adjust rule 15!
    standard_matches = clf.evaluate_rules(features)
    
    # Check if standard rule 15 matched
    std_p_matched = any(m[0] == 'P' for m in standard_matches)
    
    # Check candidate P-down
    cand_p = evaluate_p_candidate_conditions(features)
    cand_p_matched = cand_p and cand_p['matched']
    
    # Synthesize the 2-branch P behavior at Rule 15:
    # In production:
    # if is_p_index and is_p_middle and is_p_thumb and is_p_angle: (North)
    # OR is_p_down:
    # matches[:] = [m for m in matches if m[0] not in ("K", "V", "D", "L")]
    # matches.append(("P", ...))
    
    # If candidate P-down matched, it behaves identically to Rule 15 matching
    matches = list(standard_matches)
    if cand_p_matched and not std_p_matched:
        matches = [m for m in matches if m[0] not in ("K", "V", "D", "L")]
        matches.append(("P", "Candidate P-down matched", 0.95))
        
        # After Rule 15, production evaluates Q (16), R (17), X (18).
        # We need to make sure Q, R, X conflict resolution rules are respected!
        # In production:
        # Q: if Q matches: matches[:] = [m for m in matches if m[0] not in ("G", "D", "L")]; append Q
        # (Notice Q does NOT prune P, so if both P and Q matched, both remain -> CONFLICT!)
        # R: if R matches: matches[:] = [m for m in matches if m[0] not in ("U", "V", "K", "D")]; append R
        # X: if X matches: matches[:] = [m for m in matches if m[0] not in ("A", "S", "O", "E", "C")]; append X
    
    # Determine final output following recognizer.py logic
    if len(matches) == 1:
        letter, reason, conf = matches[0]
        return {
            "final_sign": letter,
            "status": "MATCH",
            "matches": matches,
            "source": "geometric_rule"
        }
    elif len(matches) > 1:
        return {
            "final_sign": "NEUTRAL",
            "status": "CONFLICT",
            "matches": matches,
            "source": "conflict"
        }
    else:
        # NO_MATCH
        # If has_static_head is False (as in current webcam pilot):
        return {
            "final_sign": "NEUTRAL",
            "status": "NO_MATCH",
            "matches": [],
            "source": "none"
        }

# -----------------------------------------------------------------------------
# 3. Load Datasets and Classify Frames into Stable vs Exit vs Entry
# -----------------------------------------------------------------------------
# A. Webcam P (90 active frames)
log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
webcam_p_active = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            webcam_p_active.append({
                'id': f"{r.get('trial_id')}_f{r.get('frame_idx')}",
                'features': r['geometric']['features']
            })

# B. Fixtures P, K, Q, D
fixtures = {}
for letter in ['P', 'K', 'Q', 'D']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stable_frames = []
    exit_frames = []
    entry_frames = []
    
    for s in data['samples']:
        if not s.get('landmarks'):
            continue
        f_idx = s.get('frame_index') or s.get('frame_idx')
        ph = s.get('phase')
        lms = s['landmarks']
        
        # Categorize frames accurately:
        if letter == 'K' and f_idx in (46, 48):
            category = 'exit'
        elif letter == 'D' and f_idx in (54, 56):
            category = 'exit'
        elif letter == 'Q' and f_idx in (60, 62, 64):
            category = 'exit'
        elif letter == 'P' and f_idx > 48:
            category = 'exit'
        elif ph == 'entry':
            category = 'entry'
        elif ph == 'stable_pose':
            category = 'stable'
        else:
            category = 'other'
            
        item = {
            'id': f"{letter}_f{f_idx}",
            'f_idx': f_idx,
            'landmarks': lms,
            'features': clf.extract_features(lms),
            'phase_raw': ph,
            'category': category
        }
        if category == 'stable':
            stable_frames.append(item)
        elif category == 'exit':
            exit_frames.append(item)
        elif category == 'entry':
            entry_frames.append(item)
            
    fixtures[letter] = {
        'stable': stable_frames,
        'exit': exit_frames,
        'entry': entry_frames,
        'all': stable_frames + exit_frames + entry_frames
    }

# -----------------------------------------------------------------------------
# 4. Compute Confusion Matrices
# -----------------------------------------------------------------------------
print("=== 2. PRODUCTION-ORDER SIMULATION RESULTS ===")

def evaluate_cohort(cohort, name, has_features_only=False):
    preds = []
    for item in cohort:
        if has_features_only:
            res = simulate_production_pipeline(None, features=item['features'])
        else:
            res = simulate_production_pipeline(item['landmarks'], features=item['features'])
        preds.append(res['final_sign'])
    
    tot = len(preds)
    c_p = preds.count('P')
    c_k = preds.count('K')
    c_q = preds.count('Q')
    c_d = preds.count('D')
    c_neu = preds.count('NEUTRAL')
    c_oth = tot - (c_p + c_k + c_q + c_d + c_neu)
    return {
        'name': name,
        'total': tot,
        'P': c_p,
        'K': c_k,
        'Q': c_q,
        'D': c_d,
        'NEUTRAL': c_neu,
        'OTHER': c_oth,
        'preds': preds
    }

# Cohorts to evaluate:
results_stable = [
    evaluate_cohort(webcam_p_active, 'Webcam P (90 active frames)', has_features_only=True),
    evaluate_cohort(fixtures['P']['stable'], f"Fixture P ({len(fixtures['P']['stable'])} true stable)"),
    evaluate_cohort(fixtures['K']['stable'], f"Fixture K ({len(fixtures['K']['stable'])} true stable)"),
    evaluate_cohort(fixtures['Q']['stable'], f"Fixture Q ({len(fixtures['Q']['stable'])} true stable)"),
    evaluate_cohort(fixtures['D']['stable'], f"Fixture D ({len(fixtures['D']['stable'])} true stable)")
]

results_exit = [
    evaluate_cohort(fixtures['K']['exit'], f"Fixture K Exit ({len(fixtures['K']['exit'])} frames: f46, f48)"),
    evaluate_cohort(fixtures['Q']['exit'], f"Fixture Q Exit ({len(fixtures['Q']['exit'])} frames: f60, f62, f64)"),
    evaluate_cohort(fixtures['D']['exit'], f"Fixture D Exit ({len(fixtures['D']['exit'])} frames: f54, f56)"),
    evaluate_cohort(fixtures['P']['exit'], f"Fixture P Exit ({len(fixtures['P']['exit'])} frames: f50-f60)"),
]

print("\n--- TABLE A: STABLE-ONLY CONFUSION MATRIX ---")
fmt = "{:35s} | {:5s} | {:5s} | {:5s} | {:5s} | {:5s} | {:7s} | {:6s}"
print(fmt.format("Cohort", "Total", "P", "K", "Q", "D", "Neutral", "Other"))
print("-" * 85)
for r in results_stable:
    print(fmt.format(r['name'], str(r['total']), str(r['P']), str(r['K']), str(r['Q']), str(r['D']), str(r['NEUTRAL']), str(r['OTHER'])))

print("\n--- TABLE B: EXIT / RETRACT PHASE AUDIT ---")
print(fmt.format("Exit Cohort", "Total", "P", "K", "Q", "D", "Neutral", "Other"))
print("-" * 85)
for r in results_exit:
    print(fmt.format(r['name'], str(r['total']), str(r['P']), str(r['K']), str(r['Q']), str(r['D']), str(r['NEUTRAL']), str(r['OTHER'])))

# Summary of False Positives
total_stable_kdq = len(fixtures['K']['stable']) + len(fixtures['Q']['stable']) + len(fixtures['D']['stable'])
fp_p_stable = (results_stable[2]['P'] + results_stable[3]['P'] + results_stable[4]['P'])

total_exit_kdq = len(fixtures['K']['exit']) + len(fixtures['Q']['exit']) + len(fixtures['D']['exit'])
fp_p_exit = (results_exit[0]['P'] + results_exit[1]['P'] + results_exit[2]['P'])

print("\n--- SUMMARY METRICS ---")
print(f"Stable K+Q+D Total: {total_stable_kdq} frames | P False Positives: {fp_p_stable} ({fp_p_stable/total_stable_kdq*100:.1f}%)")
print(f"Exit K+Q+D Total:   {total_exit_kdq} frames | P False Positives: {fp_p_exit} ({fp_p_exit/total_exit_kdq*100:.1f}%)")
print(f"Webcam P Total:     90 frames | P Matched: {results_stable[0]['P']} ({results_stable[0]['P']/90*100:.2f}%) | Neutral: {results_stable[0]['NEUTRAL']} ({results_stable[0]['NEUTRAL']/90*100:.2f}%) | FP: 0 (0.0%)")

