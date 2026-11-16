import sys
sys.path.insert(0, '.')
import os
import json
import hashlib
import glob
import collections

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def analyze_holdout_file(filepath):
    print(f"=== ANALYZING WEBCAM HOLDOUT FILE: {filepath} ===")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None
        
    sha256_hash = compute_sha256(filepath)
    print(f"File SHA256: {sha256_hash}")
    
    # Exclude baseline pilot 044024
    if "044024" in filepath:
        print("WARNING: This is the old training/baseline pilot file (044024), NOT a holdout!")
        
    frames = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                frames.append(rec)
            except Exception as e:
                print(f"Error parsing line {line_no}: {e}")
                
    print(f"Total frames loaded: {len(frames)}")
    if not frames:
        print("No frames found!")
        return None
        
    # Group by trial
    trials = collections.defaultdict(list)
    for f in frames:
        t_id = f.get('trial_id') or 'unknown_trial'
        trials[t_id].append(f)
        
    print(f"Total trials found: {len(trials)} ({list(trials.keys())})")
    
    # Inspect model status
    first_frame = frames[0]
    sh_info = first_frame.get('static_head', {})
    print(f"Model static_head load_status: {sh_info.get('load_status')}, has_static_head: {first_frame.get('has_static_head')}")
    
    # Classify frames into Stable Hold vs Transition/Exit
    # In a 5.0s trial at ~30 FPS (or whatever FPS):
    # Frames where hand is held in P pose vs entry/exit
    stable_frames = []
    transition_frames = []
    
    for t_id, t_frames in trials.items():
        n = len(t_frames)
        # If trial has frames, identify hold vs boundary
        # A standard trial: entry is ~10-15%, exit is ~15-20%
        # Or check if frame has explicit phase or kinematic features
        for i, f in enumerate(t_frames):
            # Record relative position in trial
            rel_pos = i / max(1, n - 1)
            f['_rel_pos'] = rel_pos
            f['_trial_frame_idx'] = i
            
            # Check geometric feature availability
            geom = f.get('geometric', {})
            feats = geom.get('features', {})
            
            # Simple heuristic for hold vs transition if no phase tag:
            # Entry: first 15% of frames or until hand stable
            # Exit: last 15% of frames when hand lowers (idx_tip_y drops or d_PW widens)
            # Or if phase is present
            if f.get('phase'):
                if f['phase'] == 'stable_pose':
                    stable_frames.append(f)
                else:
                    transition_frames.append(f)
            else:
                # Based on relative position and hand presence
                if 0.15 <= rel_pos <= 0.85:
                    stable_frames.append(f)
                else:
                    transition_frames.append(f)
                    
    print(f"\nPartitioning:")
    print(f"  Stable Hold Frames     : {len(stable_frames)}")
    print(f"  Transition/Exit Frames : {len(transition_frames)}")
    
    # A. Stable Hold Analysis
    print("\n--- A. STABLE HOLD ANALYSIS ---")
    s_tot = len(stable_frames)
    s_geom_p = sum(1 for f in stable_frames if f.get('geometric', {}).get('letter') == 'P' and f.get('geometric', {}).get('status') == 'MATCH')
    s_final_p = sum(1 for f in stable_frames if f.get('final_sign') == 'P')
    s_neutral = sum(1 for f in stable_frames if f.get('final_sign') == 'NEUTRAL')
    s_other = sum(1 for f in stable_frames if f.get('final_sign') not in ('P', 'NEUTRAL'))
    
    other_signs_s = collections.Counter(f.get('final_sign') for f in stable_frames if f.get('final_sign') not in ('P', 'NEUTRAL'))
    
    print(f"  Total Stable Frames      : {s_tot}")
    print(f"  Geometric MATCH P        : {s_geom_p} ({s_geom_p/max(1, s_tot)*100:.2f}%)")
    print(f"  Final Prediction P       : {s_final_p} ({s_final_p/max(1, s_tot)*100:.2f}%)")
    print(f"  Final NEUTRAL            : {s_neutral} ({s_neutral/max(1, s_tot)*100:.2f}%)")
    print(f"  Final Other (False Pos)  : {s_other} ({s_other/max(1, s_tot)*100:.2f}%) {dict(other_signs_s)}")
    
    # B. Transition / Exit Analysis
    print("\n--- B. TRANSITION / EXIT ANALYSIS ---")
    t_tot = len(transition_frames)
    t_geom_p = sum(1 for f in transition_frames if f.get('geometric', {}).get('letter') == 'P' and f.get('geometric', {}).get('status') == 'MATCH')
    t_final_p = sum(1 for f in transition_frames if f.get('final_sign') == 'P')
    t_confirmed_p = sum(1 for f in transition_frames if f.get('confirmed') and f.get('final_sign') == 'P')
    t_false_tokens = [f.get('new_token') for f in transition_frames if f.get('new_token') and f.get('new_token') != 'P']
    
    print(f"  Total Transition Frames  : {t_tot}")
    print(f"  Geometric MATCH P        : {t_geom_p} ({t_geom_p/max(1, t_tot)*100:.2f}%)")
    print(f"  Instantaneous Pred P     : {t_final_p} ({t_final_p/max(1, t_tot)*100:.2f}%)")
    print(f"  Confirmed P in Transition: {t_confirmed_p}")
    print(f"  False Tokens in Sentence : {len(t_false_tokens)} {t_false_tokens}")
    
    # C. Trial-Level Analysis
    print("\n--- C. TRIAL-LEVEL ANALYSIS ---")
    trial_stats = []
    p_dominant_trials = 0
    p_confirmed_trials = 0
    fp_trials = 0
    
    for t_id, t_frames in sorted(trials.items()):
        final_signs = [f.get('final_sign') for f in t_frames]
        counts = collections.Counter(final_signs)
        dom_sign, dom_count = counts.most_common(1)[0]
        has_p_confirmed = any(f.get('confirmed') and f.get('final_sign') == 'P' for f in t_frames)
        has_fp_confirmed = any(f.get('confirmed') and f.get('final_sign') not in ('P', 'NEUTRAL') for f in t_frames)
        
        if dom_sign == 'P':
            p_dominant_trials += 1
        if has_p_confirmed:
            p_confirmed_trials += 1
        if has_fp_confirmed:
            fp_trials += 1
            
        print(f"  Trial {t_id:32s}: {len(t_frames):3d} frames | Dominant: {dom_sign:7s} ({dom_count:2d}/{len(t_frames):2d}) | Confirmed P: {has_p_confirmed} | Confirmed FP: {has_fp_confirmed}")
        
    print(f"\nTrial Summary:")
    print(f"  Total Trials            : {len(trials)}")
    print(f"  P-Dominant Trials       : {p_dominant_trials} / {len(trials)} ({p_dominant_trials/max(1, len(trials))*100:.1f}%)")
    print(f"  P-Confirmed Trials      : {p_confirmed_trials} / {len(trials)} ({p_confirmed_trials/max(1, len(trials))*100:.1f}%)")
    print(f"  False Positive Trials   : {fp_trials} / {len(trials)}")
    
    return {
        'filepath': filepath,
        'sha256': sha256_hash,
        'total_frames': len(frames),
        'total_trials': len(trials),
        'stable_frames': s_tot,
        'stable_p_acc': s_final_p / max(1, s_tot),
        'transition_frames': t_tot,
        'p_confirmed_trials': p_confirmed_trials,
        'fp_trials': fp_trials
    }

if __name__ == '__main__':
    # Find newest pilot file
    diag_files = glob.glob('backend/data/diagnostics/pilot_*.jsonl')
    print("Available diagnostic files:", diag_files)
    if len(sys.argv) > 1:
        analyze_holdout_file(sys.argv[1])
    else:
        # Check if there is any file other than 044024 and 042505
        fresh_files = [f for f in diag_files if "044024" not in f and "042505" not in f]
        if fresh_files:
            newest = max(fresh_files, key=os.path.getmtime)
            analyze_holdout_file(newest)
        else:
            print("No fresh holdout file found beyond the baseline files (044024, 042505)!")

