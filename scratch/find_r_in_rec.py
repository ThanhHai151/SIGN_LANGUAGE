with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if '"R"' in line or "'R'" in line:
            print(f'{i+1}: {line.strip()}')

