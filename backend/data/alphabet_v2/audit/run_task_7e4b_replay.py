import json
import os
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.recognizer import VSLRecognizer
from backend.vision.diacritic_state_machine import DiacriticStateMachine

seq_files = {
    'Ơ': 'backend/data/alphabet_v2/motion_sequences/Ơ.json',
    'Ư': 'backend/data/alphabet_v2/motion_sequences/Ư.json',
    'Ô': 'backend/data/alphabet_v2/motion_sequences/Ô.json',
    'Ă': 'backend/data/alphabet_v2/motion_sequences/Ă.json'
}

base_groups = {
    'Ơ': 'O',
    'Ư': 'U',
    'Ô': 'O',
    'Ă': 'A'
}

competing_map = {
    'Ơ': 'I',
    'Ư': 'I',
    'Ô': None,
    'Ă': 'V'
}

rec = VSLRecognizer()
report = {}

for target, fpath in seq_files.items():
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seq = data.get('sequence', [])
    vid_id = data.get('video_id', '')
    expected_base = base_groups[target]
    competing_rule = competing_map.get(target)

    session_id = f'replay_7e4b_{target}'
    rec.reset_diacritic_fsm(session_id=session_id)

    tokens = []
    base_timeline = []
    candidate_events = []
    fsm_resets = 0
    base_stable_time = None
    base_stable_frame = None
    baseline_retained_frames = 0
    diacritic_confirmed_frames = []
    prev_had_baseline = False

    fsm = rec.diacritic_fsms.setdefault(session_id, DiacriticStateMachine(min_switch_confirm_frames=2))

    for item in seq:
        lms = item.get('landmarks')
        t = item.get('timestamp_sec', 0.0)
        idx = item.get('frame_index', 0)
        phase = item.get('phase', 'UNKNOWN')

        # Baseline check before update
        had_baseline_before = (fsm.baseline is not None)

        # Static recognition
        alpha_res = rec.alphabet_classifier.classify_detailed(lms)
        raw_static_sign = alpha_res.get('letter') if alpha_res.get('status') == 'MATCH' else 'UNKNOWN'
        raw_conf = alpha_res.get('confidence', 0.0)

        # Predict through recognizer
        res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, timestamp=t)
        fsm_curr = rec.diacritic_fsms.get(session_id)
        has_baseline_after = (fsm_curr.baseline is not None) if fsm_curr else False

        # Track resets
        if had_baseline_before and not has_baseline_after and lms:
            fsm_resets += 1

        if has_baseline_after:
            baseline_retained_frames += 1
            if base_stable_frame is None:
                base_stable_frame = idx
                base_stable_time = t

        # Track candidate events
        if fsm_curr and fsm_curr.switch_candidate_base is not None:
            candidate_events.append({
                'frame_index': idx,
                'timestamp': t,
                'switch_candidate_base': fsm_curr.switch_candidate_base,
                'switch_candidate_count': fsm_curr.switch_candidate_count,
                'current_base': fsm_curr.current_base,
                'state': fsm_curr.state
            })

        sign = res.get('sign')
        diac_detail = res.get('classifier_details', {}).get('diacritic_fsm', {})

        base_timeline.append({
            'frame_index': idx,
            'timestamp': t,
            'phase': phase,
            'raw_static': raw_static_sign,
            'raw_conf': raw_conf,
            'fsm_state': diac_detail.get('state'),
            'fsm_base': diac_detail.get('base_label'),
            'diacritic_detected': diac_detail.get('diacritic_detected'),
            'output_sign': sign,
            'source': res.get('source')
        })

        if diac_detail.get('state') == 'DIACRITIC_CONFIRMED' and diac_detail.get('diacritic_detected'):
            diacritic_confirmed_frames.append({
                'frame_index': idx,
                'timestamp': t,
                'diacritic': diac_detail.get('diacritic_detected'),
                'reason': diac_detail.get('reason')
            })

        if sign not in ('NEUTRAL', 'UNKNOWN'):
            if not tokens or tokens[-1][1] != sign:
                tokens.append((idx, sign))

    report[target] = {
        'video_id': vid_id,
        'total_frames': len(seq),
        'expected_base': expected_base,
        'competing_rule': competing_rule,
        'base_stable_frame': base_stable_frame,
        'base_stable_time': base_stable_time,
        'baseline_retained_frames': baseline_retained_frames,
        'fsm_resets': fsm_resets,
        'candidate_events_count': len(candidate_events),
        'candidate_events': candidate_events,
        'diacritic_confirmed_frames': diacritic_confirmed_frames,
        'confirmed_tokens': tokens,
        'base_timeline_sample': [f for f in base_timeline if f['raw_static'] != 'UNKNOWN' or f['diacritic_detected'] or f['frame_index'] in (42, 43, 44, 45, 46, 57)]
    }

output_path = 'backend/data/alphabet_v2/audit/task_7e4b_replay_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"SUCCESS: Replay results saved to {output_path}")
for k, v in report.items():
    print(f"[{k}] Vid {v['video_id']} | Stable: Frame {v['base_stable_frame']} (t={v['base_stable_time']}) | "
          f"Retained frames: {v['baseline_retained_frames']} | Resets: {v['fsm_resets']} | "
          f"Diac confirmed: {[d['diacritic'] for d in v['diacritic_confirmed_frames']]} | "
          f"Tokens: {v['confirmed_tokens']}")
