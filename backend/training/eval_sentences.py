"""
Đánh giá chuyển văn bản → ký hiệu trên bộ câu mẫu (backend/tests/fixtures/vsl_sentences.json)
==========================================================================================
Mỗi câu có danh sách must_sign: các từ robot PHẢI diễn bằng ký hiệu (không được đánh vần).
Chạy:  python -m backend.training.eval_sentences            (in bảng tổng hợp)
       python -m backend.training.eval_sentences --detail   (in từng câu)
"""
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict

from backend.nlp.text_to_vsl import text_to_vsl_glosses

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "vsl_sentences.json")
TOPICS = {"truong_hoc": "Trường học", "giao_tiep": "Giao tiếp cơ bản", "gia_dinh": "Gia đình"}


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").lower().replace("_", " ").split())


def is_signed(item: dict) -> bool:
    """Robot diễn bằng ký hiệu thật (chuyển động đã học hoặc tư thế vẽ sẵn), không phải đánh vần."""
    return item.get("type") != "fingerspelling" and bool(item.get("motion_id") or item.get("pose_key"))


def _sequence(text: str) -> list:
    r = text_to_vsl_glosses(text)
    return r.get("sequence", []) if isinstance(r, dict) else r


def _gloss_view(seq: list) -> list:
    """[(từ, cách diễn)] — các chữ cái đánh vần liên tiếp của cùng một từ được gộp lại thành một mục."""
    out = []
    for x in seq:
        if x.get("type") == "fingerspelling":
            word = x.get("spelled_word") or x.get("token")
            if out and out[-1][1] == "đánh vần" and out[-1][2] == word:
                continue
            out.append((word, "đánh vần", word))
        elif is_signed(x):
            out.append((x.get("token"), "ký hiệu", None))
        else:
            out.append((x.get("token"), "đánh vần", None))
    return [(t, k) for t, k, _ in out]


def check_sentence(s: dict) -> dict:
    seq = _sequence(s["text"])
    signed_tokens = [_norm(x.get("token")) for x in seq if is_signed(x)]
    covered = {syl for t in signed_tokens for syl in t.split()}   # các tiếng nằm trong một ký hiệu
    missing = []
    for w in s["must_sign"]:
        w = _norm(w)
        # đạt nếu cả từ nằm trong một ký hiệu, hoặc từng tiếng của nó đều được diễn bằng ký hiệu ("đi làm" → đi + làm)
        ok = any(w == t or f" {w} " in f" {t} " for t in signed_tokens) or all(syl in covered for syl in w.split())
        if not ok:
            missing.append(w)
    words = [x for x in seq if x.get("type") != "fingerspelling"]
    spelled_letters = [x for x in seq if x.get("type") == "fingerspelling"]
    return {
        "topic": s["topic"],
        "text": s["text"],
        "glosses": _gloss_view(seq),
        "missing": missing,
        "n_must": len(s["must_sign"]),
        "n_words": len(words),
        "n_words_signed": sum(is_signed(x) for x in words),
        "n_letters": len(spelled_letters),
    }


def evaluate(path: str = FIXTURE) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [check_sentence(s) for s in data["sentences"]]
    by_topic = defaultdict(lambda: {"sentences": 0, "ok": 0, "must": 0, "must_ok": 0})
    for r in rows:
        t = by_topic[r["topic"]]
        t["sentences"] += 1
        t["ok"] += not r["missing"]
        t["must"] += r["n_must"]
        t["must_ok"] += r["n_must"] - len(r["missing"])
    total = {k: sum(t[k] for t in by_topic.values()) for k in ("sentences", "ok", "must", "must_ok")}
    missing_words = Counter(w for r in rows for w in r["missing"])
    return {"rows": rows, "by_topic": dict(by_topic), "total": total, "missing_words": missing_words}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    res = evaluate()
    if "--detail" in argv:
        for r in res["rows"]:
            mark = "OK " if not r["missing"] else "LỖI"
            seq = " | ".join(f"{t}" + ("" if k == "ký hiệu" else f" [{k}]") for t, k in r["glosses"])
            print(f"{mark} {r['text']}\n     → {seq}")
            if r["missing"]:
                print(f"     thiếu ký hiệu: {', '.join(r['missing'])}")
    print("\nChủ đề              Câu đạt    Từ bắt buộc diễn được")
    for key, t in res["by_topic"].items():
        print(f"{TOPICS.get(key, key):<20}{t['ok']:>3}/{t['sentences']:<6}{t['must_ok']:>4}/{t['must']} "
              f"({100 * t['must_ok'] / max(1, t['must']):.1f}%)")
    t = res["total"]
    print(f"{'Tổng':<20}{t['ok']:>3}/{t['sentences']:<6}{t['must_ok']:>4}/{t['must']} "
          f"({100 * t['must_ok'] / max(1, t['must']):.1f}%)")
    if res["missing_words"]:
        print("\nTừ chưa diễn được:", ", ".join(f"{w} ({n})" for w, n in res["missing_words"].most_common()))


if __name__ == "__main__":
    main()
