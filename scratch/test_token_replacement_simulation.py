import sys
sys.path.insert(0, ".")
from scratch.test_o_patch_simulation2 import rec, dsm
import json
import collections

DIACRITIC_BASE_MAP = {
    "Ă": "A", "Â": "A", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U", "Đ": "D"
}

VIDEOS = {
    'Ă': ('backend/data/alphabet_v2/motion_sequences/Ă.json', 'A'),
    'Â': ('backend/data/alphabet_v2/motion_sequences/Â.json', 'A'),
    'Ê': ('backend/data/alphabet_v2/motion_sequences/Ê.json', 'E'),
    'Đ': ('backend/data/alphabet_v2/motion_sequences/Đ.json', 'D'),
    'Ô': ('backend/data/alphabet_v2/motion_sequences/Ô.json', 'O'),
    'Ơ': ('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'O'),
    'Ư': ('backend/data/alphabet_v2/motion_sequences/Ư.json', 'U'),
}

for target, (path, exp_base) in VIDEOS.items():
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    seq = d['sequence']
    session_id = f"test_replace_{target}"
    rec.reset()
    rec.reset_diacritic_fsm(session_id)
    
    sentence_tokens = []
    emitted_events = []
    last_confirmed_sign = None
    last_confirmed_time = 0.0
    last_base_token_time = 0.0
    last_confirmed_diacritic = None
    seen_neutral_release = True
    
    for item in seq:
        idx = item.get('frame_index')
        t = item.get('timestamp_sec', 0.0)
        lms = item.get('landmarks')
        res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, trial_id=session_id, frame_idx=idx, timestamp=t, expected_sign=target)
        
        final_sign = res.get('sign')
        
        if final_sign in ('NEUTRAL', 'UNKNOWN') or not lms:
            seen_neutral_release = True
            
        recent = [s for s, c in rec.prediction_history if c >= rec.confidence_threshold]
        if len(recent) >= rec.smoothing_window // 2:
            most_common, count = collections.Counter(recent).most_common(1)[0]
            if count >= (rec.smoothing_window // 2) and most_common != "NEUTRAL":
                is_spurious_exit = False
                if most_common == "V" and last_confirmed_diacritic == "Ă":
                    if not seen_neutral_release:
                        is_spurious_exit = True
                        
                if not is_spurious_exit and (most_common != last_confirmed_sign or (t - last_confirmed_time > rec.cooldown_sec)):
                    last_confirmed_sign = most_common
                    last_confirmed_time = t
                    
                    replaced = None
                    if most_common in DIACRITIC_BASE_MAP:
                        b = DIACRITIC_BASE_MAP[most_common]
                        if len(sentence_tokens) > 0 and sentence_tokens[-1] == b and (t - last_base_token_time <= 2.5):
                            replaced = sentence_tokens[-1]
                            sentence_tokens[-1] = most_common
                            emitted_events.append((idx, round(t, 2), most_common, f"REPLACED_{replaced}"))
                        else:
                            sentence_tokens.append(most_common)
                            emitted_events.append((idx, round(t, 2), most_common, "APPEND"))
                        last_confirmed_diacritic = most_common
                        seen_neutral_release = False
                    else:
                        sentence_tokens.append(most_common)
                        emitted_events.append((idx, round(t, 2), most_common, "APPEND"))
                        if most_common in ("A", "E", "O", "U", "D"):
                            last_base_token_time = t
                            
    print(f"{target:<3}: Events={emitted_events} | Final sentence_tokens={sentence_tokens}")
