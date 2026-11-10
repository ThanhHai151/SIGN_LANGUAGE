import json
import os
from backend.vision.recognizer import VSLRecognizer
from backend.vision.diacritic_state_machine import DiacriticStateMachine

seq_files = {
    'Ă': 'backend/data/alphabet_v2/motion_sequences/Ă.json',
    'Â': 'backend/data/alphabet_v2/motion_sequences/Â.json',
    'Ê': 'backend/data/alphabet_v2/motion_sequences/Ê.json',
    'Ô': 'backend/data/alphabet_v2/motion_sequences/Ô.json',
    'Ơ': 'backend/data/alphabet_v2/motion_sequences/Ơ.json',
    'Ư': 'backend/data/alphabet_v2/motion_sequences/Ư.json',
    'Đ': 'backend/data/alphabet_v2/motion_sequences/Đ.json'
}

base_groups = {'Ă': 'A', 'Â': 'A', 'Ê': 'E', 'Ô': 'O', 'Ơ': 'O', 'Ư': 'U', 'Đ': 'D'}

rec = VSLRecognizer()

report = {}

for target, fpath in seq_files.items():
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seq = data.get('sequence', [])
    vid_id = data.get('video_id', '')
    expected_base = base_groups[target]

    # Run current AFTER pipeline first, collecting frame-by-frame static predictions
    session_after = f'sess_{target}'
    rec.reset_diacritic_fsm(session_id=session_after)
    tokens_after = []
    after_transient_frames = 0
    after_baseline_retained_frames = 0
    after_reached_base_stable = False
    valid_lm_count = 0
    valid_base_count = 0
    phase_counts = {}

    frame_records = []

    for item in seq:
        lms = item.get('landmarks')
        t = item.get('timestamp_sec', 0.0)
        idx = item.get('frame_index', 0)
        phase = item.get('phase', 'UNKNOWN')
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

        if lms:
            valid_lm_count += 1

        res = rec.predict_landmarks(lms, mode='letters', session_id=session_after, timestamp=t)
        fsm_curr = rec.diacritic_fsms.get(session_after)

        static_details = res.get('classifier_details', {}).get('static_classification', {})
        base_sign = static_details.get('sign', res.get('sign'))
        base_conf = static_details.get('confidence', res.get('confidence', 0.0))

        frame_records.append({
            'frame_index': idx,
            'timestamp': t,
            'landmarks': lms,
            'phase': phase,
            'base_sign': base_sign,
            'base_conf': base_conf,
            'res_sign': res.get('sign'),
            'fsm_state': fsm_curr.state if fsm_curr else 'IDLE',
            'has_baseline': (fsm_curr.baseline is not None) if fsm_curr else False
        })

        if base_sign == expected_base:
            valid_base_count += 1
        elif base_sign in ('NEUTRAL', 'UNKNOWN') and lms:
            after_transient_frames += 1
            if fsm_curr and fsm_curr.baseline is not None:
                after_baseline_retained_frames += 1

        if fsm_curr and fsm_curr.state == 'BASE_STABLE':
            after_reached_base_stable = True

        sign_a = res.get('sign')
        if sign_a not in ('NEUTRAL', 'UNKNOWN'):
            if not tokens_after or tokens_after[-1][1] != sign_a:
                tokens_after.append((idx, sign_a))

    # BEFORE SIMULATION: exactly how recognizer behaved before Task 7E.3
    # where if base_sign not in diacritic_base_groups -> fsm.reset()
    fsm_before = DiacriticStateMachine()
    tokens_before = []
    before_resets_on_neutral = 0
    before_reached_base_stable = False

    DIACRITIC_BASE_GROUPS = ("A", "E", "O", "U", "D")
    for rec_item in frame_records:
        lms = rec_item['landmarks']
        t = rec_item['timestamp']
        idx = rec_item['frame_index']
        base_sign = rec_item['base_sign']
        base_conf = rec_item['base_conf']

        if base_sign in DIACRITIC_BASE_GROUPS:
            fsm_res = fsm_before.update(base_sign, base_conf, lms, timestamp=t)
            sign_b = fsm_res['final_label']
        else:
            if fsm_before.baseline is not None and base_sign in ('NEUTRAL', 'UNKNOWN') and lms:
                before_resets_on_neutral += 1
            fsm_before.reset()
            sign_b = 'NEUTRAL' if base_sign in ('NEUTRAL', 'UNKNOWN') else base_sign

        if fsm_before.state == 'BASE_STABLE':
            before_reached_base_stable = True

        if sign_b not in ('NEUTRAL', 'UNKNOWN'):
            if not tokens_before or tokens_before[-1][1] != sign_b:
                tokens_before.append((idx, sign_b))

    report[target] = {
        'video_id': vid_id,
        'total_frames': len(seq),
        'valid_landmarks': valid_lm_count,
        'phase_counts': phase_counts,
        'valid_base_frames': valid_base_count,
        'before': {
            'reached_base_stable': before_reached_base_stable,
            'resets_on_neutral': before_resets_on_neutral,
            'confirmed_tokens': tokens_before
        },
        'after': {
            'reached_base_stable': after_reached_base_stable,
            'transient_neutral_frames': after_transient_frames,
            'baseline_retained_frames': after_baseline_retained_frames,
            'confirmed_tokens': tokens_after
        }
    }

output_path = 'backend/data/alphabet_v2/audit/task_7e3_replay_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print('SUCCESS: Replay written to', output_path)
for k, v in report.items():
    tok_bef = v['before']['confirmed_tokens']
    resets_bef = v['before']['resets_on_neutral']
    tok_aft = v['after']['confirmed_tokens']
    retained_aft = v['after']['baseline_retained_frames']
    print(f"[{k}] Vid {v['video_id']} | Before: {tok_bef} (resets: {resets_bef}) | After: {tok_aft} (retained bl frames: {retained_aft})")
