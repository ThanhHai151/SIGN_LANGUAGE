import json
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.models.vsl_dictionary import ALL_SIGNS, VSL_ALPHABET

for let in ['T', 'U', 'S']:
    print(f"ALL_SIGNS.get('{let}'): {ALL_SIGNS.get(let)}")
    print(f"'{let}' in VSL_ALPHABET: {let in VSL_ALPHABET}")

