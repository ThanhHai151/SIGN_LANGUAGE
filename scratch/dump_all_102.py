import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# Read source files
import importlib.util, os

# Load the dictionary module
spec = importlib.util.spec_from_file_location("vsl_102", "backend/models/vsl_102_dictionary.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SIGN_DICT_102 = mod.SIGN_DICT_102
print(f"Total classes: {len(SIGN_DICT_102)}")
print()

# Print all 102 entries
for idx, entry in enumerate(SIGN_DICT_102):
    print(f"{idx}: {json.dumps(entry, ensure_ascii=False)}")

