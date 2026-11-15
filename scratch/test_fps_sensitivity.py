import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from backend.vision.recognizer import VSLRecognizer
from backend.vision.diacritic_state_machine import DiacriticStateMachine

seq_files = {
    'Ă': 'backend/data/alphabet_v2/motion_sequences/Ă.json',
    'Â': 'backend/data/alphabet_v2/motion_sequences/Â.json',
    'Ê': 'backend/data/alphabet_v2/motion_sequences/Ê.json',
    'Đ': 'backend/data/alphabet_v2/motion_sequences/Đ.json',
    'Ô': 'backend/data/alphabet_v2/motion_sequences/Ô.json',
    'Ơ': 'backend/data/alphabet_v2/motion_sequences/Ơ.json',
    'Ư': 'backend/data/alphabet_v2/motion_sequences/Ư.json',
}

def test_fps_with_base_frames(base_frames, fps_target=10):
    rec = VSLRecognizer()
    min_dt = 1.0 / fps_target
    confirmed_count = 0
    
    print(f"\n--- Testing with min_base_stable_frames={base_frames} at {fps_target} FPS ---")
    for letter, fpath in seq_files.items():
        seq = json.load(open(fpath, encoding='utf-8'))['sequence']
        sub_seq = []
        last_t = -1.0
        for item in seq:
            t = item.get('timestamp_sec') or 0.0
            if last_t < 0 or (t - last_t) >= (min_dt * 0.95):
                sub_seq.append(item)
                last_t = t
                
        sid = f'replay_{fps_target}fps_{letter}'
        rec.reset_diacritic_fsm(sid)
        rec.diacritic_fsms[sid] = DiacriticStateMachine(
            min_base_stable_frames=base_frames,
            min_base_stable_sec=0.10,
            max_diacritic_search_sec=1.50,
            max_diacritic_search_frames=35
        )
        
        confirmed = False
        tokens = []
        for item in sub_seq:
            lms = item.get('landmarks')
            t = item.get('timestamp_sec') or 0.0
            res = rec.predict_landmarks(lms, mode='letters', session_id=sid, timestamp=t)
            sign = res.get('sign')
            fsm_det = res.get('classifier_details', {}).get('diacritic_fsm', {})
            if fsm_det.get('state') == 'DIACRITIC_CONFIRMED' and fsm_det.get('diacritic_detected'):
                confirmed = True
            if sign not in ('NEUTRAL', 'UNKNOWN') and (not tokens or tokens[-1] != sign):
                tokens.append(sign)
                
        if confirmed: confirmed_count += 1
        print(f"  {letter}: Confirmed={confirmed} | Tokens={tokens}")
        
    print(f"Total confirmed at {fps_target} FPS with base_frames={base_frames}: {confirmed_count}/7")
    return confirmed_count

test_fps_with_base_frames(4, 10)
test_fps_with_base_frames(3, 10)
test_fps_with_base_frames(2, 10)
test_fps_with_base_frames(3, 15)
test_fps_with_base_frames(2, 15)

