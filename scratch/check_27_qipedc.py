import sys
sys.path.insert(0, '.')
import json
from backend.models.vsl_102_dictionary import SIGN_DICT_102
from backend.models.lexicon import load_lexicon

lex = load_lexicon()
gt_indices = {39, 40, 41, 43, 44, 45, 50, 51, 54, 55, 56, 57, 58, 62, 65, 66, 68, 72, 73, 75, 76, 77, 79, 80, 81, 82, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100}
q_indices = [i for i in range(39, 102) if i not in gt_indices]
print(f'Total QIPEDC classes in 39..101: {len(q_indices)}')
for i in q_indices:
    term = SIGN_DICT_102.get(i)
    by_sstcn = lex['by_sstcn'].get(i)
    eid = by_sstcn['id'] if by_sstcn else 'NONE'
    reg = by_sstcn.get('region') if by_sstcn else 'NONE'
    vid = by_sstcn.get('video_url') if by_sstcn else 'NONE'
    print(f'  i={i:3d}: term={term:<15} eid={eid:<18} reg={reg:<5} vid={vid}')
