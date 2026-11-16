import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Ensure venv site-packages are loaded
venv_site = r"D:\SIGN_LANGUAGE\.venv\Lib\site-packages"
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# Ensure project root is in sys.path
proj_root = r"D:\SIGN_LANGUAGE"
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import pytest

print("Running test suite with pytest...")
exit_code = pytest.main(["backend/tests", "-v"])
sys.exit(exit_code)

