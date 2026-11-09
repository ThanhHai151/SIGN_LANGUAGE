import re

for filepath in ['backend/training/train_model.py', 'backend/training/run_alphabet_extraction.py', 'backend/training/run_alphabet_v2_extraction.py']:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f"=== {filepath} ===")
            for line in text.splitlines()[:30]:
                print(line)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")

