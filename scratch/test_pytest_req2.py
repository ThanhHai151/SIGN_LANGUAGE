import sys, os
sys.path.insert(0, os.path.abspath("."))
import pytest

with open('backend/vision/diacritic_state_machine.py', 'r', encoding='utf-8') as f:
    orig = f.read()

patched = orig.replace('req_conf_frames = 1 if detected_letter == "Đ" else 2', 'req_conf_frames = 2')
with open('backend/vision/diacritic_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(patched)

try:
    ret1 = pytest.main(['backend/tests/test_diacritic_state_machine.py', '-k', 'test_case_8'])
    ret2 = pytest.main(['backend/tests/test_static_head_fallback.py', '-k', 'test_d5'])
    print("test_case_8 exit code:", ret1)
    print("test_d5 exit code:", ret2)
finally:
    with open('backend/vision/diacritic_state_machine.py', 'w', encoding='utf-8') as f:
        f.write(orig)

