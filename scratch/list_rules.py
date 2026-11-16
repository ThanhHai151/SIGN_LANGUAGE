import re

with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

matches = re.findall(r'matches\.append\(\(\s*["\']([A-ZĂÂĐÊÔƠƯ])["\']', text)
unique_matches = sorted(list(set(matches)))
print(f"Letters with geometric rules in evaluate_rules ({len(unique_matches)}):", unique_matches)
