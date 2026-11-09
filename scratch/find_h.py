with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if '"H"' in line or "'H'" in line:
            print(f'{i+1}: {line.strip()}')

