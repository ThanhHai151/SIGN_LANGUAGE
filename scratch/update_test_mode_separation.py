# -*- coding: utf-8 -*-
with open('backend/tests/test_mode_separation.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = 'self.assertEqual(res3["sentence"], "D Số 3")'
replacement = 'self.assertEqual(res3["sentence"], "D 3")'
assert target in text, 'target not found'
text = text.replace(target, replacement, 1)

with open('backend/tests/test_mode_separation.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated test_mode_separation.py successfully')

