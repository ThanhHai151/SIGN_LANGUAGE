import json
import os
from backend.vision.recognizer import VSLRecognizer
from backend.vision.diacritic_state_machine import DiacriticStateMachine

seq_files = {
    'Ơ': 'backend/data/alphabet_v2/motion_sequences/Ơ.json',
    'Ư': 'backend/data/alphabet_v2/motion_sequences/Ư.json',
    'Ă': 'backend/data/alphabet_v2/motion_sequences/Ă.json'
}

base_groups = {'Ơ': 'O', 'Ư': 'U', 'Ă': 'A'}
competing_map = {'Ơ': 'I', 'Ư': 'I', 'Ă': 'V'}

rec = VSLRecognizer()

report = {}

for target, fpath in seq_files.items():
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seq = data.get('sequence', [])
    vid_id = data.get('video_id', '')
    expected_base = base_groups[target]
    competing_rule = competing_map[target]

    session_id = f'replay_7e4a_{target}'
    rec.reset_diacritic_fsm(session_id=session_id)

    tokens = []
    rule_competing_matches = 0
    fsm_resets = 0
    baseline_lost_count = 0
    fsm_retained_frames = 0
    diacritic_confirmed_frames = []
    competing_overwritten_frames = []

    fsm = rec.diacritic_fsms.setdefault(session_id, DiacriticStateMachine())
    prev_had_baseline = False

    for item in seq:
        lms = item.get('landmarks')
        t = item.get('timestamp_sec', 0.0)
        idx = item.get('frame_index', 0)
        phase = item.get('phase', 'UNKNOWN')

        # Static prediction before FSM
        alpha_res = rec.alphabet_classifier.classify_detailed(lms)
        raw_static_sign = alpha_res.get('letter') if alpha_res.get('status') == 'MATCH' else 'UNKNOWN'
        if raw_static_sign == competing_rule:
            rule_competing_matches += 1

        res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, timestamp=t)
        fsm_curr = rec.diacritic_fsms.get(session_id)
        current_has_baseline = (fsm_curr.baseline is not None) if fsm_curr else False

        if prev_had_baseline and not current_has_baseline and lms:
            baseline_lost_count += 1
            fsm_resets += 1
        elif current_has_baseline:
            fsm_retained_frames += 1

        prev_had_baseline = current_has_baseline

        sign = res.get('sign')
        diac_detail = res.get('classifier_details', {}).get('diacritic_fsm', {})
        if diac_detail.get('state') == 'DIACRITIC_CONFIRMED' and diac_detail.get('diacritic_detected'):
            diacritic_confirmed_frames.append((idx, diac_detail.get('diacritic_detected')))

        if sign == competing_rule and diac_detail.get('source') != 'diacritic_fsm':
            competing_overwritten_frames.append((idx, sign))

        if sign not in ('NEUTRAL', 'UNKNOWN'):
            if not tokens or tokens[-1][1] != sign:
                tokens.append((idx, sign))

    report[target] = {
        'video_id': vid_id,
        'total_frames': len(seq),
        'expected_base': expected_base,
        'competing_rule': competing_rule,
        'rule_competing_matches': rule_competing_matches,
        'baseline_lost_count': baseline_lost_count,
        'fsm_resets': fsm_resets,
        'fsm_retained_frames': fsm_retained_frames,
        'diacritic_confirmed_frames': diacritic_confirmed_frames,
        'competing_overwritten_frames': competing_overwritten_frames,
        'confirmed_tokens': tokens
    }

output_path = 'backend/data/alphabet_v2/audit/task_7e4a_replay_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print('SUCCESS: Replay written to', output_path)
for k, v in report.items():
    print(f"[{k}] Vid {v['video_id']} | Tokens: {v['confirmed_tokens']} | Confirmed diac frames: {v['diacritic_confirmed_frames']} | Overwritten: {v['competing_overwritten_frames']} | Retained frames: {v['fsm_retained_frames']}")
