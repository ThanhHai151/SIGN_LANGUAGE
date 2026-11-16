import sys
sys.path.insert(0, '.')
import os
import json
import collections
from backend.vision.recognizer import VSLRecognizer

def run_production_replay():
    # 1. Instantiate the production recognizer
    recognizer = VSLRecognizer()
    
    print("=== PRODUCTION RECOGNIZER INITIALIZATION STATE ===")
    print(f"has_static_head        : {recognizer.has_static_head}")
    print(f"static_head_load_status: {recognizer.static_head_load_status}")
    print(f"static_head_load_error : {recognizer.static_head_load_error}")
    print(f"confidence_threshold   : {recognizer.confidence_threshold}")
    print(f"smoothing_window       : {recognizer.smoothing_window}")
    print(f"cooldown_sec           : {recognizer.cooldown_sec}")
    from backend.vision.recognizer import ALLOWED_LETTERS
    print(f"allowed_letters count  : {len(ALLOWED_LETTERS)}")
    
    # 2. Load X.json
    fixture_path = 'backend/data/alphabet_v2/static_landmarks/X.json'
    with open(fixture_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    samples = data.get('samples', [])
    valid_samples = [s for s in samples if s.get('landmarks')]
    print(f"\nTotal valid samples in X.json: {len(valid_samples)}")
    
    # Replay all frames in strict chronological order
    # Session ID to ensure clean session state
    session_id = "test_replay_task1_4_1"
    
    # Track per-frame results
    all_results = []
    
    for idx, s in enumerate(valid_samples):
        f_idx = s.get('frame_index') or s.get('frame_idx') or idx
        ts = s.get('timestamp_sec')
        phase = s.get('phase', 'unknown')
        lms = s['landmarks']
        
        # State BEFORE processing
        history_before = list(recognizer.prediction_history)
        last_confirmed_before = recognizer.last_confirmed_sign
        last_confirmed_time_before = recognizer.last_confirmed_time
        
        # Direct geometric classification call for observation
        geom_res = recognizer.alphabet_classifier.classify_detailed(lms)
        
        # Actual production recognizer call!
        res = recognizer.predict_landmarks(
            landmarks_list=lms,
            expected_sign="X",
            session_id=session_id,
            mode="letters",
            timestamp=ts,
            frame_idx=f_idx
        )
        
        # State AFTER processing
        history_after = list(recognizer.prediction_history)
        recent_signs = [sign for sign, conf in history_after if conf >= recognizer.confidence_threshold]
        vote_counts = collections.Counter(recent_signs)
        
        all_results.append({
            'frame_index': f_idx,
            'timestamp': ts,
            'phase': phase,
            'geom_status': geom_res.get('status'),
            'geom_letter': geom_res.get('letter'),
            'geom_conf': geom_res.get('confidence'),
            'geom_reason': geom_res.get('reason'),
            'runtime_sign': res.get('sign'),
            'runtime_conf': res.get('confidence'),
            'runtime_source': res.get('source'),
            'runtime_confirmed': res.get('confirmed'),
            'runtime_new_token': res.get('new_token'),
            'last_confirmed_sign': recognizer.last_confirmed_sign,
            'last_confirmed_time': recognizer.last_confirmed_time,
            'history_after': history_after,
            'recent_signs': recent_signs,
            'vote_counts': dict(vote_counts),
        })

    # Save all results to scratch json
    with open('scratch/task1_4_1_replay_all_frames.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved {len(all_results)} frame results to scratch/task1_4_1_replay_all_frames.json")
    
    # Focus table on frames 134 to 148
    print("\n=== FOCUS ANALYSIS: FRAMES 134 to 148 (CRITICAL TRANSITION & EXIT) ===")
    focus_frames = [r for r in all_results if r['frame_index'] and 134 <= r['frame_index'] <= 148]
    
    header = (
        f"{'Frame':6s} | {'Time':6s} | {'Phase':11s} | "
        f"{'Geom':10s} | {'Pre-Smooth':10s} | {'Source':15s} | "
        f"{'History / Buffer':32s} | {'Votes':18s} | "
        f"{'Confirmed?':10s} | {'Last Confirmed':14s}"
    )
    print(header)
    print("-" * len(header))
    
    for r in focus_frames:
        f_str = f"{r['frame_index']}"
        t_str = f"{r['timestamp']:.3f}" if r['timestamp'] else "N/A"
        ph_str = f"{r['phase']}"
        g_str = f"{r['geom_letter'] or 'NONE'} ({r['geom_status'][:4]})"
        ps_str = f"{r['runtime_sign']} ({r['runtime_conf']:.2f})"
        src_str = f"{r['runtime_source']}"
        hist_str = str([s for s, c in r['history_after']])
        if len(hist_str) > 32:
            hist_str = hist_str[:29] + "..."
        votes_str = str(r['vote_counts'])
        conf_str = "CONFIRMED" if r['runtime_confirmed'] else "no"
        last_str = f"{r['last_confirmed_sign']}"
        
        print(f"{f_str:6s} | {t_str:6s} | {ph_str:11s} | {g_str:10s} | {ps_str:10s} | {src_str:15s} | {hist_str:32s} | {votes_str:18s} | {conf_str:10s} | {last_str:14s}")

if __name__ == '__main__':
    run_production_replay()
