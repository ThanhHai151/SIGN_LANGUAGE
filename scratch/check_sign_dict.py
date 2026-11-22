import sys
sys.path.insert(0, '.')
from backend.models.vsl_102_dictionary import SIGN_DICT_102
print('Total in SIGN_DICT_102:', len(SIGN_DICT_102))
user_gt_indices = {39, 40, 41, 43, 44, 45, 50, 51, 54, 55, 56, 57, 58, 62, 65, 66, 68, 72, 73, 75, 76, 77, 79, 80, 81, 82, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100}
print(f'Count of GT indices: {len(user_gt_indices)}')
for idx in sorted(user_gt_indices):
    print(f'  idx={idx}: term="{SIGN_DICT_102.get(idx)}"')

remaining_indices = [i for i in range(102) if i not in user_gt_indices and i >= 37]
print(f'Remaining word indices (>=37 and not GT): {len(remaining_indices)}')
for idx in remaining_indices:
    print(f'  idx={idx}: term="{SIGN_DICT_102.get(idx)}"')
