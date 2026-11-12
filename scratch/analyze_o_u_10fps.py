import os
import sys
import json
sys.path.insert(0, os.path.abspath("."))

for let, fpath in [("Ơ", "backend/data/alphabet_v2/motion_sequences/Ơ.json"), ("Ư", "backend/data/alphabet_v2/motion_sequences/Ư.json")]:
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    seq = data["sequence"]
    
    print(f"\n=== LETTER {let} ({fpath}) ===")
    print(f"Total frames: {len(seq)}")
    
    # Check timestamps and frames where landmarks are present
    valid_seq = [s for s in seq if s.get("landmarks")]
    print(f"Valid frames: {len(valid_seq)}")
    
    # Subsample at 10 FPS (dt = 0.1s)
    seq_10fps = []
    last_t = -1.0
    for s in seq:
        t = float(s.get("timestamp_sec", 0.0))
        if last_t < 0 or (t - last_t) >= 0.095:
            seq_10fps.append(s)
            last_t = t
    print(f"Frames at 10fps: {len(seq_10fps)}")
    
    # Let's see what base sign is recognized for each frame in 10fps
    from backend.vision.recognizer import VSLRecognizer
    rec = VSLRecognizer()
    rec.reset_diacritic_fsm(f"check_{let}")
    
    for idx, s in enumerate(seq_10fps):
        lms = s.get("landmarks")
        t = float(s.get("timestamp_sec", 0.0))
        f_orig = s.get("frame_index")
        
        # Check classifier detailed directly
        geom_res = rec.alphabet_classifier.classify_detailed(lms) if lms else None
        
        res = rec.predict_landmarks(lms, mode="letters", session_id=f"check_{let}", timestamp=t)
        fsm = rec.diacritic_fsms.get(f"check_{let}")
        
        state = fsm.state if fsm else "None"
        base = fsm.current_base if fsm else "None"
        base_cnt = fsm.candidate_base_count if fsm else 0
        cand = fsm.diacritic_candidate if fsm else "None"
        cand_cnt = fsm.diacritic_candidate_count if fsm else 0
        
        print(f"  [{idx:02d}] orig_f={f_orig:02d} t={t:.3f}s: geom={geom_res['letter'] if geom_res else 'None'} ({geom_res['status'] if geom_res else 'None'}) -> rec_sign={res['sign']} | fsm_state={state} base={base} (cnt={base_cnt}) cand={cand} (cnt={cand_cnt})")
