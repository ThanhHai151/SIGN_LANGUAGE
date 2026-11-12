import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()
seq_files = {
    'Ă': 'backend/data/alphabet_v2/motion_sequences/Ă.json',
    'Â': 'backend/data/alphabet_v2/motion_sequences/Â.json',
    'Ê': 'backend/data/alphabet_v2/motion_sequences/Ê.json',
    'Đ': 'backend/data/alphabet_v2/motion_sequences/Đ.json',
    'Ô': 'backend/data/alphabet_v2/motion_sequences/Ô.json',
    'Ơ': 'backend/data/alphabet_v2/motion_sequences/Ơ.json',
    'Ư': 'backend/data/alphabet_v2/motion_sequences/Ư.json',
}

target_fps_list = [None, 10, 5, 3, 2]

results_fps = {}

for fps in target_fps_list:
    fps_key = 'native' if fps is None else f'{fps}fps'
    results_fps[fps_key] = {}
    
    for letter, fpath in seq_files.items():
        seq = json.load(open(fpath, encoding='utf-8'))['sequence']
        
        if fps is None:
            sub_seq = seq
        else:
            min_dt = 1.0 / fps
            sub_seq = []
            last_t = -1.0
            for item in seq:
                t = item.get('timestamp_sec') or 0.0
                if last_t < 0 or (t - last_t) >= (min_dt * 0.95):
                    sub_seq.append(item)
                    last_t = t
        
        sid = f'replay_{fps_key}_{letter}'
        rec.reset_diacritic_fsm(sid)
        
        base_stable_f = None
        motion_detected = False
        confirmed = False
        confirmed_frame = None
        confirmed_time = None
        tokens = []
        
        for item in sub_seq:
            lms = item.get('landmarks')
            t = item.get('timestamp_sec') or 0.0
            idx = item.get('frame_index', 0)
            
            res = rec.predict_landmarks(lms, mode='letters', session_id=sid, timestamp=t)
            fsm = rec.diacritic_fsms.get(sid)
            fsm_det = res.get('classifier_details', {}).get('diacritic_fsm', {})
            sign = res.get('sign')
            
            if fsm and fsm.baseline and base_stable_f is None:
                base_stable_f = idx
            if fsm_det.get('state') == 'DIACRITIC_CANDIDATE':
                motion_detected = True
            if fsm_det.get('state') == 'DIACRITIC_CONFIRMED' and fsm_det.get('diacritic_detected'):
                confirmed = True
                if confirmed_frame is None:
                    confirmed_frame = idx
                    confirmed_time = t
            if sign not in ('NEUTRAL', 'UNKNOWN') and (not tokens or tokens[-1] != sign):
                tokens.append(sign)
                
        results_fps[fps_key][letter] = {
            'total_sub_frames': len(sub_seq),
            'base_stable_frame': base_stable_f,
            'motion_detected': motion_detected,
            'confirmed': confirmed,
            'confirmed_frame': confirmed_frame,
            'confirmed_time': round(confirmed_time, 3) if confirmed_time else None,
            'tokens': tokens
        }

print("=== LOW-FPS EVALUATION SUMMARY ===")
for fps_key in ['native', '10fps', '5fps', '3fps', '2fps']:
    print(f"-- Rate: {fps_key} --")
    for letter in ['Ă', 'Â', 'Ê', 'Đ', 'Ơ', 'Ư', 'Ô']:
        r = results_fps[fps_key][letter]
        st = 'CONFIRMED' if r['confirmed'] else 'NO'
        c_frame = r['confirmed_frame']
        c_time = r['confirmed_time']
        cf_str = f"f{c_frame} ({c_time}s)" if r['confirmed'] else "None"
        bs_str = str(r['base_stable_frame'])
        frames_cnt = r['total_sub_frames']
        tok_str = str(r['tokens'])
        print(f"  {letter}: {st:<9} | frame: {cf_str:<18} | base_stable: {bs_str:<4} | frames: {frames_cnt:<3} | tokens: {tok_str}")
