import numpy as np
import json

ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
print("Keys:", ds.files)
meta_str = str(ds['metadata_json'])
meta = json.loads(meta_str)
print("Meta type:", type(meta))
print("Meta keys:", list(meta.keys()) if isinstance(meta, dict) else len(meta))
if isinstance(meta, dict):
    for k in list(meta.keys())[:10]:
        print(f"  {k}: {meta[k]}")
elif isinstance(meta, list):
    print("First item in meta:", meta[0])
    # check letters in meta
    letters = {}
    for item in meta:
        l = item.get('letter') or item.get('letter_label')
        letters[l] = letters.get(l, 0) + 1
    print("Letters in metadata:", letters)

