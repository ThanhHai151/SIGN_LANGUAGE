import json
import sys
sys.path.insert(0, ".")

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

motion_files = {
    "Ă": "backend/data/alphabet_v2/motion_sequences/Ă.json",
    "Â": "backend/data/alphabet_v2/motion_sequences/Â.json",
    "Ê": "backend/data/alphabet_v2/motion_sequences/Ê.json",
    "Đ": "backend/data/alphabet_v2/motion_sequences/Đ.json",
    "Ô": "backend/data/alphabet_v2/motion_sequences/Ô.json",
    "Ơ": "backend/data/alphabet_v2/motion_sequences/Ơ.json",
    "Ư": "backend/data/alphabet_v2/motion_sequences/Ư.json",
}

for target, fpath in motion_files.items():
    with open(fpath, "r", encoding="utf-8") as f:
        seq = json.load(f).get("sequence", [])
        
    session_id = f"trace_tokens_{target}"
    rec.reset()
    rec.reset_diacritic_fsm(session_id)
    
    emitted = []
    
    for item in seq:
        lms = item.get("landmarks")
        t = item.get("timestamp_sec", 0.0)
        idx = item.get("frame_index", 0)
        
        res = rec.predict_landmarks(lms, mode="letters", session_id=session_id, timestamp=t)
        if res.get("confirmed") and res.get("new_token"):
            emitted.append({
                "frame": idx,
                "time": round(t, 3),
                "token": res.get("new_token"),
                "sign": res.get("sign"),
                "source": res.get("source"),
                "pred_source": res.get("prediction_source"),
                "history": list(rec.prediction_history)[-5:]
            })
            
    print(f"\nTarget: {target} (Total frames: {len(seq)})")
    for e in emitted:
        hist_str = ", ".join([f"{s}" for s, _ in e['history']])
        print(f"  -> Frame {e['frame']:02d} ({e['time']:.2f}s): Token='{e['token']}' (source={e['source']}) | Recent: [{hist_str}]")

