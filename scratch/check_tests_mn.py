import glob

tests = glob.glob('backend/tests/*.py')
for t in tests:
    with open(t, 'r', encoding='utf-8') as f:
        content = f.read()
    if "'M'" in content or '"M"' in content or "'N'" in content or '"N"' in content:
        lines = [l.strip() for l in content.splitlines() if any(k in l for k in ["'M'", '"M"', "'N'", '"N"'])][:3]
        print(f'{t}:')
        for l in lines:
            print(f'   {l}')

