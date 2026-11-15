import json

with open('backend/data/diagnostics/automated_validation_frames_20261112_231703.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('target') == 'Ă' and item.get('rep') == 1:
            idx = item.get('frame_index')
            if idx >= 25 and idx <= 50:
                print(f"f{idx:02d} (t={item['timestamp_sec']:.3f}): sign={item['predicted_sign']:<4} src={item['prediction_source']:<20} conf={item['confidence']} fsm_state={item['fsm_state']} tok={item['new_token']}")

