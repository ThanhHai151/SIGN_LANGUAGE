import json

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    lex = json.load(f)

entries = lex['entries']
gt_in_lex = [e for e in entries if e['id'].startswith('gt-')]
print(f'Total entries: {len(entries)}')
print(f'GT in lex count: {len(gt_in_lex)}')

user_gt = {
    'gt-thoi-gian', 'gt-hoi-gio', 'gt-ngay', 'gt-nam', 'gt-hom-nay', 'gt-hom-qua', 'gt-dem',
    'gt-toi', 'gt-xin-phep', 'gt-vui', 'gt-buon', 'gt-met', 'gt-khoe', 'gt-tuoi', 'gt-o-dau', 'gt-lam-gi',
    'gt-dung', 'gt-nhay', 'gt-ngu', 'gt-an', 'gt-uong', 'gt-tam', 'gt-cuoi',
    'gt-gia-dinh', 'gt-ong', 'gt-ba', 'gt-vo', 'gt-chong', 'gt-anh-trai', 'gt-chi-gai', 'gt-yeu',
    'gt-cong-nhan', 'gt-nong-dan', 'gt-bac-si', 'gt-y-ta', 'gt-cong-an'
}
lex_gt_ids = {e['id'] for e in gt_in_lex}
print(f'Diff lex - user: {lex_gt_ids - user_gt}')
print(f'Diff user - lex: {user_gt - lex_gt_ids}')

# Check QIPEDC entries with same Vietnamese words
for e in gt_in_lex:
    w = e['word']
    q_matches = [q['id'] for q in entries if q['word'].lower() == w.lower() and not q['id'].startswith('gt-')]
    print(f'  {e["id"]} (word="{w}", sstcn={e.get("sstcn_index")}) -> QIPEDC equivalents: {q_matches}')
