import os

for f in ['scratch/test_tuned.py', 'scratch/test_refined_rule.py', 'scratch/test_n_rule_candidates.py', 'scratch/test_o_rule_candidates.py']:
    if os.path.exists(f):
        print(f"=== {f} ===")
        with open(f, 'r', encoding='utf-8') as fp:
            for l in fp.readlines()[:30]:
                print(l, end='')

