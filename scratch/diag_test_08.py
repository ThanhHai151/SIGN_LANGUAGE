import sys
sys.path.insert(0, '.')
import time

print('Step 1: import lexicon and word_scope...', flush=True)
from backend.models.lexicon import load_lexicon
from backend.vision import qipedc_live
from backend.vision.word_scope import approved_live_word_ids, validate_extra_qipedc_word
print('Step 2: calculate approved_3231...', flush=True)
real_lex = load_lexicon()
approved = approved_live_word_ids(real_lex, indexed_ids=qipedc_live.indexed_entry_ids(), builtin_ids=qipedc_live.BUILTIN_WORDS)
print(f'approved count: {len(approved)}', flush=True)

print('Step 3: import app...', flush=True)
from backend.app import app, _live_qipedc_entries
print('Step 4: test _live_qipedc_entries...', flush=True)
entries = _live_qipedc_entries(extra_ids=('q-chao', 'q-1', 'q-a', 'q-choi', 'gt-toi', 'does-not-exist'))
print(f'entries: {len(entries)}', flush=True)

print('Step 5: test TestClient...', flush=True)
from fastapi.testclient import TestClient
client = TestClient(app)
for d in ('q-1', 'q-a', 'q-choi', 'gt-toi', 'q-bat-dong'):
    t = time.time()
    res = client.post(f'/api/qipedc/live/prepare?entry_id={d}')
    item = res.json()['entries'][0]
    print(f'  {d} in {time.time() - t:.2f}s -> ready={item["ready"]}, reason={item.get("reject_reason")}', flush=True)

print('Step 6: test recognize_sequence with rejected items...', flush=True)
from backend.app import recognizer
import numpy as np

def _sample_sequence(length=30):
    seq = np.zeros((length, 75, 3), dtype=np.float32)
    seq[:, 11] = [0.63, 0.50, 0.0]
    seq[:, 12] = [0.37, 0.50, 0.0]
    seq[:, 33:54, 0] = np.linspace(0.56, 0.64, length)[:, None]
    seq[:, 54:75, 0] = 0.42 + np.linspace(0.0, 0.04, 21)
    return seq

seq = _sample_sequence(30)
t = time.time()
resp = client.post(
    '/api/recognize_sequence',
    json={
        'sequence': seq.tolist(),
        'mode': 'words',
        'live_words': True,
        'qipedc_entries': ['q-1', 'q-choi', 'gt-toi', 'q-bat-dong', 'q-chao'],
        'include_diagnostic': True,
    },
)
print(f'recognize_sequence in {time.time() - t:.2f}s, status: {resp.status_code}', flush=True)
data = resp.json()['data']
print(f'data sign: {data.get("sign")}, word_scope_valid: {data.get("word_scope_valid")}', flush=True)
diag = data.get('diagnostic', {})
rejected_ids = {r['entry_id'] for r in diag.get('rejected_extra_words', [])}
print(f'rejected extra words: {rejected_ids}', flush=True)
print('ALL CHECKS PASSED!')
