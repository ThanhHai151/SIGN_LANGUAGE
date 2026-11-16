import json

with open('backend/data/diagnostics/automated_validation_frames_20261112_231703.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('target') == 'Ô' and item.get('rep') == 1:
            idx = item.get('frame_index')
            sign = item.get('predicted_sign')
            src = item.get('prediction_source')
            fsm_st = item.get('fsm_state')
            fsm_b = item.get('fsm_base')
            tok = item.get('new_token')
            if sign != 'NEUTRAL' or tok or fsm_st not in ['IDLE', None]:
                print(f"Frame {idx:02d}: sign={sign}, src={src}, fsm_state={fsm_st}, fsm_base={fsm_b}, token={tok}")

