import sys
sys.path.insert(0, '.')
import json
import hashlib
from fastapi.testclient import TestClient
from backend.models.lexicon import load_lexicon

load_lexicon.cache_clear()
real_lex = load_lexicon()

print('=== 1. DATA INTEGRITY AUDIT ===')
with open('backend/data/vsl_lexicon.json.bak', 'r', encoding='utf-8') as f:
    bak_lex = json.load(f)

cur_entries = real_lex['entries']
bak_entries = bak_lex['entries']

print(f'Backup total entries: {len(bak_entries)}')
print(f'Current total entries: {len(cur_entries)}')
assert len(bak_entries) == 3363, f"Expected 3363, got {len(bak_entries)}"
assert len(cur_entries) == 3327, f"Expected 3327, got {len(cur_entries)}"

# Check GT entries
cur_gt = [e for e in cur_entries if e['id'].startswith('gt-')]
bak_gt = [e for e in bak_entries if e['id'].startswith('gt-')]
print(f'Backup GT count: {len(bak_gt)}')
print(f'Current GT count: {len(cur_gt)}')
assert len(bak_gt) == 36
assert len(cur_gt) == 0

# Check that every single non-GT entry in backup is preserved in current
bak_non_gt_ids = {e['id'] for e in bak_entries if not e['id'].startswith('gt-')}
cur_ids = {e['id'] for e in cur_entries}
print(f'Preserved QIPEDC entries: {len(cur_ids)} / {len(bak_non_gt_ids)}')
assert cur_ids == bak_non_gt_ids, f"Diff: {bak_non_gt_ids - cur_ids}"

# Check Supabase URLs and GT regions in current
cur_supa = [e['id'] for e in cur_entries if 'supabase.co' in (e.get('video_url') or '') or any('supabase.co' in (v.get('url') or '') for v in e.get('variants', []))]
cur_gt_reg = [e['id'] for e in cur_entries if e.get('region') == 'GT' or any(v.get('region') == 'GT' for v in e.get('variants', []))]
print(f'Current entries with Supabase URL: {len(cur_supa)}')
print(f'Current entries with region==GT: {len(cur_gt_reg)}')
assert len(cur_supa) == 0
assert len(cur_gt_reg) == 0

print('\n=== 2. CHECKPOINT & MODEL AUDIT ===')
from backend.vision.recognizer import VSLRecognizer
from backend.vision.word_scope import allowed_sstcn_word_indices, LEGACY_GT_SSTCN_INDICES

rec = VSLRecognizer()
model = rec.sstcn_model
num_classes = getattr(model, 'num_classes', 102)
print(f'SSTCN num_classes: {num_classes}')
assert num_classes == 102

active_idx, meta = allowed_sstcn_word_indices(model, real_lex)
print(f'Active SSTCN word classes: {len(active_idx)}')
gt_ex = [x for x in meta["excluded"] if x["reason"] == "unverified_gt_curriculum"]
print(f'Excluded GT classes: {len(gt_ex)}')
assert len(active_idx) == 27, f"Expected 27, got {len(active_idx)}"
assert len(gt_ex) == 36, f"Expected 36, got {len(gt_ex)}"

print('\n=== 3. API ENDPOINTS AUDIT ===')
from backend.app import app
client = TestClient(app)

# /api/lexicon
r_lex = client.get('/api/lexicon')
assert r_lex.status_code == 200
r_lex_j = r_lex.json()
print(f'/api/lexicon total: {r_lex_j["total"]}')
assert r_lex_j['total'] == 3327
assert len([e for e in r_lex_j['entries'] if e['id'].startswith('gt-')]) == 0

# /api/lexicon/gt-toi (must be 404)
r_gt = client.get('/api/lexicon/gt-toi')
print(f'/api/lexicon/gt-toi status: {r_gt.status_code}')
assert r_gt.status_code == 404

# /api/sstcn/topics
r_top = client.get('/api/sstcn/topics')
assert r_top.status_code == 200
r_top_j = r_top.json()
print(f'/api/sstcn/topics word_classes_count: {r_top_j["word_classes_count"]}')
print(f'/api/sstcn/topics excluded_gt_count: {r_top_j["excluded_gt_count"]}')
assert r_top_j['word_classes_count'] == 27
assert r_top_j['excluded_gt_count'] == 36
assert all(not (w.get('entry_id') or '').startswith('gt-') for w in r_top_j['words'])

print('\n=== 4. TEXT-TO-SIGN TRANSLATION AUDIT ===')
from backend.nlp.text_to_vsl import text_to_vsl_glosses
test_sentences = [
    "Tôi ăn cơm",
    "Xin chào bạn",
    "Gia đình tôi có ông bà bố mẹ",
    "Bác sĩ và y tá làm việc ở bệnh viện"
]
for sent in test_sentences:
    tokens = text_to_vsl_glosses(sent)
    for tok in tokens:
        eid = str(tok.get('id') or '')
        vurl = str(tok.get('video_url') or '')
        assert not eid.startswith('gt-'), f"Leaked {eid} in '{sent}'"
        assert 'supabase.co' not in vurl, f"Leaked supabase in '{sent}'"
    print(f'  "{sent}" -> {[t.get("word") for t in tokens]} (all clean)')

print('\nALL VERIFICATIONS PASSED 100%!')
