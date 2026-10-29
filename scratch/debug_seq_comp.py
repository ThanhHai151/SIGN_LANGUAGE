import sys
import os
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("."))
venv_site = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

from backend.vision.recognizer import VSLRecognizer

with open('backend/tests/fixtures/real_digit_landmarks.json', 'r', encoding='utf-8') as f:
    d_data = json.load(f)
lm_1 = next(s['landmarks'] for s in d_data['samples'] if s['digit'] == '1')

with open('backend/tests/fixtures/real_digit_0_landmarks.json', 'r', encoding='utf-8') as f:
    d0_data = json.load(f)
lm_0 = d0_data['samples'][0]['landmarks']

rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.5)
t = 1000.0

for _ in range(12):
    res = rec.predict_landmarks(lm_1, mode='digits', timestamp=t)
    t += 0.05
print("After 1:", res['sign'], rec.digit_aggregator.accumulated_digits)

for _ in range(4):
    res = rec.predict_landmarks(None, mode='digits', timestamp=t)
    t += 0.05
print("After neutral:", res['sign'], rec.digit_aggregator.accumulated_digits)

for i in range(12):
    res = rec.predict_landmarks(lm_0, mode='digits', timestamp=t)
    print(f"Frame 0 #{i}: sign={res['sign']}, raw={rec.prediction_history[-1]}, agg_digits={rec.digit_aggregator.accumulated_digits}, cand={rec.digit_aggregator.current_candidate}, seen={rec.digit_aggregator.candidate_first_seen}")
    t += 0.05

print("Final after 0:", res['sign'], rec.digit_aggregator.accumulated_digits)
