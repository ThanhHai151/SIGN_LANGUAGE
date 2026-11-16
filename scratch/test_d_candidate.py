import sys, os
sys.path.insert(0, os.path.abspath("."))
import json, numpy as np

with open('backend/vision/diacritic_state_machine.py', encoding='utf-8') as f:
    code = f.read()

# Create req_conf_frames = 2 class
code_req2 = code.replace('req_conf_frames = 1 if detected_letter == "Đ" else 2', 'req_conf_frames = 2')
ns1 = {}
exec(code, ns1)
FSM1 = ns1['DiacriticStateMachine']

ns2 = {}
exec(code_req2, ns2)
FSM2 = ns2['DiacriticStateMachine']

seq_d = json.load(open('backend/data/alphabet_v2/motion_sequences/Đ.json', encoding='utf-8'))['sequence']

for name, FSMCls in [("req_conf_frames=1 (baseline)", FSM1), ("req_conf_frames=2 (candidate)", FSM2)]:
    fsm = FSMCls()
    confs = []
    cands = []
    for item in seq_d:
        res = fsm.update('D', 0.95, item.get('landmarks'), timestamp=item.get('timestamp_sec'))
        if res.get('state') == 'DIACRITIC_CANDIDATE':
            cands.append((item.get('frame_index'), item.get('timestamp_sec')))
        if res.get('state') == 'DIACRITIC_CONFIRMED':
            confs.append((item.get('frame_index'), item.get('timestamp_sec'), res.get('final_label')))
    print(f"\n{name} in D0495:")
    print("  Candidates:", cands)
    print("  Confirmed:", confs)

# Also test on Sample 525 held 10 frames
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
lm525 = npz['landmarks'][525].tolist()

for name, FSMCls in [("req_conf_frames=1", FSM1), ("req_conf_frames=2", FSM2)]:
    fsm = FSMCls()
    t = 1.0
    states = []
    for i in range(10):
        r = fsm.update('D', 0.95, lm525, timestamp=t)
        states.append(r['final_label'])
        t += 0.04
    print(f"{name} on Sample 525 (10 frames): all D? {all(s == 'D' for s in states)}")

# Also test on Low-FPS sub-sampling for D0495
for fps in [None, 10, 5, 3, 2]:
    fps_label = 'native' if fps is None else f'{fps}fps'
    if fps is None:
        sub_seq = seq_d
    else:
        min_dt = 1.0 / fps
        sub_seq = []
        last_t = -1.0
        for item in seq_d:
            t = item.get('timestamp_sec') or 0.0
            if last_t < 0 or (t - last_t) >= (min_dt * 0.95):
                sub_seq.append(item)
                last_t = t
    
    for name, FSMCls in [("req=1", FSM1), ("req=2", FSM2)]:
        fsm = FSMCls()
        c_frames = []
        for item in sub_seq:
            res = fsm.update('D', 0.95, item.get('landmarks'), timestamp=item.get('timestamp_sec'))
            if res.get('state') == 'DIACRITIC_CONFIRMED':
                c_frames.append((item.get('frame_index'), item.get('timestamp_sec')))
        print(f"FPS {fps_label:<6} | {name:<5}: confirmed={len(c_frames)} {c_frames[:1]}")

# Synthetic noise test
d_indices = [i for i, target in enumerate(npz['y']) if npz['classes'][target] == 'D']
idx = d_indices[0]
lm_d = npz['landmarks'][idx]
for name, FSMCls in [("req=1", FSM1), ("req=2", FSM2)]:
    np.random.seed(42)
    fsm = FSMCls()
    false_confirms = 0
    t = 1.0
    for frame_idx in range(100):
        noisy_lm = (lm_d + np.random.normal(0, 0.006, lm_d.shape)).tolist()
        res = fsm.update('D', 0.95, noisy_lm, timestamp=t)
        if res.get('state') == 'DIACRITIC_CONFIRMED':
            false_confirms += 1
        t += 0.04
    print(f"Synthetic noise (100 frames) | {name}: false_confirms = {false_confirms}")

