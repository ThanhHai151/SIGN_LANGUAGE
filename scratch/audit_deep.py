import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

def audit():
    print("================== 1. QIPEDC DICTIONARY & LEXICON ==================")
    q_dict_path = Path("backend/data/sources/qipedc_dictionary.json")
    if q_dict_path.exists():
        q_data = json.loads(q_dict_path.read_text(encoding="utf-8"))
        print(f"qipedc_dictionary.json rows: {len(q_data)}")
        ids = [row[0] for row in q_data]
        words = [row[1] for row in q_data]
        print(f"Sample items: {q_data[:3]}")
        print(f"Unique IDs: {len(set(ids))}, Unique words: {len(set(words))}")
    
    from backend.models.lexicon import load_lexicon
    lex = load_lexicon()
    print("lexicon keys:", list(lex.keys()))
    entries = lex["entries"]
    print(f"lexicon.py entries: {len(entries)}")
    by_id = lex.get("by_id", {})
    words = set(e["word"] for e in entries)
    print(f"lexicon unique by_id: {len(by_id)}, unique words: {len(words)}")
    with_video = [e for e in entries if e.get("video_url")]
    print(f"lexicon entries with video_url: {len(with_video)}")

    print("\n================== 2. LIVE INDEX (live_index.npz) ==================")
    idx_path = Path("backend/data/qipedc_reference/live_index.npz")
    if idx_path.exists():
        idx = np.load(idx_path, allow_pickle=True)
        uni_ids = idx["uni_ids"]
        bi_ids = idx["bi_ids"]
        print(f"uni_ids: {len(uni_ids)}")
        print(f"bi_ids: {len(bi_ids)}")
        print(f"Total templates in live_index: {len(uni_ids) + len(bi_ids)}")
        uni_features = idx["uni_features"]
        bi_features = idx["bi_features"]
        print(f"uni_features shape: {uni_features.shape}")
        print(f"bi_features shape: {bi_features.shape}")
        uni_var_to_entry = idx["uni_var_to_entry"]
        bi_var_to_entry = idx["bi_var_to_entry"]
        print(f"uni variants: {len(uni_var_to_entry)}, bi variants: {len(bi_var_to_entry)}")

    print("\n================== 3. MODELS & WEIGHTS ==================")
    # Check alphabet head
    alpha_path = Path("backend/models/alphabet_static_head_v1.pt")
    if alpha_path.exists():
        import torch
        ckpt = torch.load(alpha_path, map_location="cpu", weights_only=False)
        print(f"alphabet_static_head_v1.pt classes ({len(ckpt.get('classes', []))}): {ckpt.get('classes')}")
        print(f"num_classes: {ckpt.get('num_classes')}, input_dim: {ckpt.get('input_dim')}")

    # Check sstcn models
    sstcn_path = Path("backend/models/weights/sign_sstcn_attention_model.pth")
    if sstcn_path.exists():
        ckpt = torch.load(sstcn_path, map_location="cpu", weights_only=False)
        print(f"sign_sstcn_attention_model.pth keys: {list(ckpt.keys())}")
        sd = ckpt["model_state_dict"]
        # Find final classification layer weight
        for k in reversed(list(sd.keys())):
            if "weight" in k:
                print(f"Final layer weight: {k} -> shape {sd[k].shape}")
                break

    # Check vsl_102_dictionary
    try:
        from backend.models.vsl_102_dictionary import VSL_102_CLASSES, SIGN_DICT_102
        print(f"VSL_102_CLASSES count: {len(VSL_102_CLASSES)}")
        print(f"SIGN_DICT_102 count: {len(SIGN_DICT_102)}")
        print(f"Sample VSL_102: {VSL_102_CLASSES[:10]}")
    except Exception as e:
        print("Error checking vsl_102:", e)

    # Check words_716
    w716_meta = Path("backend/data/words_716/manifest.json")
    if w716_meta.exists():
        m = json.loads(w716_meta.read_text(encoding="utf-8"))
        print(f"words_716 manifest classes: {len(m.get('classes', []))}")
    else:
        print("words_716 manifest not found or untracked")

    # Check checkpoints in words_716_checkpoints
    ckpt_703 = Path("backend/models/words_716_checkpoints/experiments_703/best.pth")
    if ckpt_703.exists():
        ckpt = torch.load(ckpt_703, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        for k in reversed(list(sd.keys())):
            if "weight" in k:
                print(f"words_716_checkpoints 703 best.pth final layer: {k} -> shape {sd[k].shape}")
                break

    print("\n================== 4. VSL RECOGNIZER INITIALIZATION ==================")
    from backend.vision.recognizer import VSLRecognizer
    rec = VSLRecognizer()
    print("VSLRecognizer initialized successfully!")
    print(f"Active mode: {getattr(rec, 'mode', None)}")
    print(f"Has SSTCN model: {hasattr(rec, 'sstcn_model') and rec.sstcn_model is not None}")
    if hasattr(rec, 'sstcn_model') and rec.sstcn_model is not None:
        print(f"SSTCN model class: {rec.sstcn_model.__class__.__name__}")
    print(f"SSTCN topics/classes count: {len(getattr(rec, 'sstcn_classes', []))}")
    if hasattr(rec, 'sstcn_classes'):
        print(f"Sample sstcn_classes: {rec.sstcn_classes[:10]}")
    print(f"Checkpoint info: {rec.get_sstcn_checkpoint_info()}")

    print("\n================== 5. QIPEDC LIVE LOGIC ==================")
    from backend.vision import qipedc_live
    print(f"qipedc_live DEFAULT_WORDS count: {len(qipedc_live.DEFAULT_WORDS)}")
    print(f"Sample DEFAULT_WORDS: {qipedc_live.DEFAULT_WORDS[:5]}")
    indexed_ids = qipedc_live.indexed_entry_ids()
    print(f"qipedc_live indexed_entry_ids count: {len(indexed_ids)}")
    print(f"Sample indexed_entry_ids: {indexed_ids[:10]}")


if __name__ == "__main__":
    audit()
