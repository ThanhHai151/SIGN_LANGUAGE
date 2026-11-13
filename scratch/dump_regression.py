"""Dump output nhận diện với dữ liệu cố định để so sánh trước/sau patch. Dùng: python dump_regression.py out.json"""
import sys, json, io, contextlib
import numpy as np

rng = np.random.default_rng(123)
out = {}
with contextlib.redirect_stdout(io.StringIO()):
    from backend.vision.recognizer import VSLRecognizer
    rec = VSLRecognizer()

    seqs = [(rng.random((30, 75, 3)) * 0.8).astype(np.float32).tolist() for _ in range(5)]
    for mode in ["words", "sentence", None]:
        rec_m = VSLRecognizer()
        out[f"sequence/{mode}"] = [rec_m.predict_sequence(s, threshold=0.65, mode=mode) for s in seqs]

    lms = [(rng.random((21, 3))).tolist() for _ in range(6)]
    for mode in ["letters", "digits", "sentence", "letters_numbers", None]:
        rec_m = VSLRecognizer()
        res = []
        for lm in lms:
            try:
                res.append(rec_m.predict_landmarks(landmarks_list=lm, multi_hands=None, expected_sign=None, mode=mode))
            except Exception as e:
                res.append({"error": type(e).__name__})
        out[f"landmarks/{mode}"] = res

json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True, default=str)
print("dumped", {k: len(v) for k, v in out.items()})

