import sys
sys.path.insert(0, '.')
import subprocess
import tempfile
import sys
import numpy as np

ws_code = subprocess.check_output(['git', 'show', '4a3c61e9:backend/vision/word_scope.py'], text=True, encoding='utf-8')
with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
    f.write(ws_code)
    tmp_path = f.name

import importlib.util
spec = importlib.util.spec_from_file_location('word_scope_temp', tmp_path)
ws = importlib.util.module_from_spec(spec)
sys.modules['word_scope_temp'] = ws
spec.loader.exec_module(ws)

from backend.models.lexicon import load_lexicon
lex = load_lexicon()

eligible_map = ws.eligible_qipedc_word_entries(lex)
print('eligible_map count:', len(eligible_map))

data = np.load('backend/data/qipedc_reference/live_index.npz')
index_3240 = set(str(x) for x in data['uni_ids']) | set(str(x) for x in data['bi_ids'])
print('live_index.npz entries count:', len(index_3240))

approved = ws.approved_live_word_ids(lex, indexed_ids=index_3240)
print('approved_live_word_ids count:', len(approved))

diff = index_3240 - set(eligible_map.keys())
print(f'In index but not eligible (count={len(diff)}):')
for eid in sorted(diff):
    entry = lex['by_id'].get(eid)
    reason = ws.is_non_word_entry(entry)
    v_info = ws.filter_entry_eligible_videos(entry)
    word = entry.get('word') if entry else None
    print(f'  {eid}: word={word}, non_word_reason={reason}, eligible={v_info.get("eligible")}')

# Where did 3,231 come from?
# Let's inspect test_words_scope_filter.py from 4a3c61e9
test_code = subprocess.check_output(['git', 'show', '4a3c61e9:backend/tests/test_words_scope_filter.py'], text=True, encoding='utf-8')
for line in test_code.splitlines():
    if '3231' in line or '3,231' in line or '3240' in line:
        print('Test mention:', line)
