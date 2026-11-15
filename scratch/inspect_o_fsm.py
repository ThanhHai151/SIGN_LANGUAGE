import json
import sys
sys.path.insert(0, ".")

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()
rec.debug_diacritic = True

with open('backend/data/alphabet_v2/motion_sequences/Ô.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

seq = data.get('sequence', [])
session_id = 'trace_o_d0507b'
rec.reset_diacritic_fsm(session_id)

print(f"{'Idx':<5} | {'Time':<6} | {'Sign':<6} | {'Src':<18} | {'FSM State':<20} | {'FSM Cand':<10} | {'Dy':<8} | {'Lift':<8} | {'Reason'}")
print("-" * 110)

for item in seq:
    lms = item.get('landmarks')
    t = item.get('timestamp_sec', 0.0)
    idx = item.get('frame_index', 0)
    
    res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, timestamp=t)
    sign = res.get('sign')
    src = res.get('source')
    
    fsm = rec.diacritic_fsms.get(session_id)
    fsm_state = fsm.state if fsm else "None"
    fsm_cand = fsm.diacritic_candidate if fsm else "None"
    lift = f"{fsm.last_lift:.4f}" if fsm and hasattr(fsm, 'last_lift') and fsm.last_lift is not None else "None"
    
    curr_y = lms[0][1] if lms else None
    base_y = fsm.baseline.wrist_y if (fsm and fsm.baseline) else None
    dy = (base_y - curr_y) if (curr_y is not None and base_y is not None) else None
    dy_str = f"{dy:.4f}" if dy is not None else "None"
    
    diag = res.get('classifier_details', {}).get('diacritic_fsm', {})
    reason = diag.get('reason', '')
    
    if idx in range(10, 45):
        print(f"{idx:<5} | {t:<6.3f} | {str(sign):<6} | {str(src):<18} | {str(fsm_state):<20} | {str(fsm_cand):<10} | {str(dy_str):<8} | {str(lift):<8} | {reason}")
