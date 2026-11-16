import os
import sys
import json
import time

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.recognizer import VSLRecognizer

VIDEOS = {
    'Ă': ('backend/data/alphabet_v2/motion_sequences/Ă.json', 'A', 16),
    'Â': ('backend/data/alphabet_v2/motion_sequences/Â.json', 'A', 34),
    'Ê': ('backend/data/alphabet_v2/motion_sequences/Ê.json', 'E', 14),
    'Đ': ('backend/data/alphabet_v2/motion_sequences/Đ.json', 'D', 42),
    'Ô': ('backend/data/alphabet_v2/motion_sequences/Ô.json', 'O', 18),
    'Ơ': ('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'O', 38),
    'Ư': ('backend/data/alphabet_v2/motion_sequences/Ư.json', 'U', 25),
}

HOLD_INTERVALS_MS = [500, 800, 1100, 1200, 1500, 2000]

# Monkey-patch or test directly
rec = VSLRecognizer()

print("=" * 100)
print("TESTING 2.5s TOKEN REPLACEMENT WINDOW WITH 1.2s TTS TIMEOUT")
print("=" * 100)

for ch, (path, exp_base, base_frame_idx) in VIDEOS.items():
    with open(path, encoding='utf-8') as f:
        seq = json.load(f)['sequence']

    print(f"\n--- Target: {ch} (Base: {exp_base}) ---")

    for hold_ms in HOLD_INTERVALS_MS:
        hold_sec = hold_ms / 1000.0
        trial_id = f"test_{ch}_{hold_ms}ms"
        rec.reset()
        rec.reset_diacritic_fsm(trial_id)

        pre_frames = [item for item in seq if item['frame_index'] <= base_frame_idx]
        post_frames = [item for item in seq if item['frame_index'] > base_frame_idx]

        new_seq = []
        for item in pre_frames:
            new_seq.append(item)

        last_t = pre_frames[-1]['timestamp_sec']
        hold_steps = int(round(hold_sec / 0.04))
        for step in range(1, hold_steps + 1):
            new_seq.append({
                'frame_index': base_frame_idx + step,
                'timestamp_sec': round(last_t + step * 0.04, 4),
                'landmarks': pre_frames[-1]['landmarks']
            })

        shift = hold_steps * 0.04
        for item in post_frames:
            new_seq.append({
                'frame_index': item['frame_index'] + hold_steps,
                'timestamp_sec': round(item['timestamp_sec'] + shift, 4),
                'landmarks': item['landmarks']
            })

        tts_events = []
        token_events = []
        replacement_events = []

        for item in new_seq:
            idx = item['frame_index']
            t = item['timestamp_sec']
            lms = item['landmarks']

            res = rec.predict_landmarks(
                lms,
                mode='letters',
                session_id=trial_id,
                trial_id=trial_id,
                frame_idx=idx,
                timestamp=t,
                expected_sign=ch
            )

            tok = res.get('new_token')
            rep = res.get('replaced_token')
            tts = res.get('tts_speak')

            if rep:
                replacement_events.append((idx, round(t, 3), rep, tok))
            if tok:
                token_events.append((idx, round(t, 3), tok, rep))
            if tts:
                tts_events.append((idx, round(t, 3), tts))

        final_sentence = rec.sentence_tokens.copy()
        spoken = [ev[2] for ev in tts_events]

        print(f"  Hold {hold_ms:>4}ms: Sentence={final_sentence} | Spoken TTS={spoken} | Rep Events={len(replacement_events)}")

