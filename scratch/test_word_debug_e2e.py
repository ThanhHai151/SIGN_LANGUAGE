"""E2E test cho Word Debug. Chạy khi backend đang chạy ở 127.0.0.1:8000."""
import json, os, glob, shutil, sys
import urllib.request, urllib.error
import numpy as np

BASE = "http://127.0.0.1:8000"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS_DIR = os.path.join(ROOT, "debug", "words")
rng = np.random.default_rng(0)
fails = []


def call(method, path, body=None, headers=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails.append(name)


def seq(right_zero=False):
    a = rng.random((30, 75, 3)).astype(np.float32) * 0.8
    if right_zero:
        a[:, 54:75, :] = 0
    return a.tolist()


def predict(mode, **kw):
    return call("POST", "/api/recognize_sequence", {"sequence": seq(**kw), "threshold": 0.65, "mode": mode})


before = set(glob.glob(os.path.join(WORDS_DIR, "*")))

# 1. targets
st, t = call("GET", "/api/debug/word/targets")
targets = t["targets"]
check("63 targets", t["total"] == 63 and len(targets) == 63)
check("class range 39..101", [x["class"] for x in targets] == list(range(39, 102)))
nam = [x for x in targets if x["word"] == "Năm"]
check("'Năm' có đúng 1 class", len(nam) == 1, str(nam))
nam_class = nam[0]["class"]

# 2. Không có phiên => predict words không tạo file
predict("words")
after_idle = set(glob.glob(os.path.join(WORDS_DIR, "*")))
check("không có phiên => không tạo log", after_idle == before)

# 3. validate
check("class ngoài 39..101 bị từ chối", call("POST", "/api/debug/word/session", {"action": "start", "target_class": 5, "trial": 1})[0] == 400)
check("từ không khớp class bị từ chối", call("POST", "/api/debug/word/session", {"action": "start", "target_class": nam_class, "target_word": "Ngày", "trial": 1})[0] == 400)
check("trial=0 bị từ chối", call("POST", "/api/debug/word/session", {"action": "start", "target_class": nam_class, "trial": 0})[0] == 400)

# 4. Start trial 1
st, s1 = call("POST", "/api/debug/word/session", {"action": "start", "target_class": nam_class, "target_word": "Năm", "trial": 1})
check("start ok", st == 200 and s1["active"] and s1["target_word"] == "Năm" and s1["target_class"] == nam_class and s1["trial"] == 1, str(s1))
sid1 = s1["session_id"]
check("tên thư mục sanitize", sid1.startswith("Nam_trial_01_"), sid1)

res = [predict("words", right_zero=(i == 0))[1] for i in range(4)]
check("predict vẫn trả success", all(r["success"] for r in res))
check("class dự đoán nằm trong 39..101", all(39 <= r["data"]["index"] <= 101 or r["data"]["index"] == 102 for r in res))
predict("sentence")   # mode khác không được ghi
predict("letters") if False else None
st, ev_ok = call("POST", "/api/debug/word/event", {"event_type": "SEQUENCE_RESET", "frames_before_reset": 17, "reason": "no_hands_detected"})
check("SEQUENCE_RESET được ghi", ev_ok["success"] is True)
check("event_type lạ bị bỏ qua", call("POST", "/api/debug/word/event", {"event_type": "PREDICTION"})[1]["success"] is False)

st, stt = call("GET", "/api/debug/word/status")
check("status đếm 4 lượt", stt["seq_count"] == 4 and stt["active"], str(stt))

# 5. Stop
st, stop = call("POST", "/api/debug/word/session", {"action": "stop"})
check("stop ok", st == 200 and stop["session_id"] == sid1)
predict("words")
check("sau Stop không ghi tiếp", call("GET", "/api/debug/word/status")[1]["active"] is False)
check("event sau Stop bị bỏ qua", call("POST", "/api/debug/word/event", {"event_type": "SEQUENCE_RESET"})[1]["success"] is False)

d1 = os.path.join(WORDS_DIR, sid1)
events = [json.loads(l) for l in open(os.path.join(d1, "events.jsonl"), encoding="utf-8")]
types = [e["event_type"] for e in events]
print("event_types:", types)
check("có SESSION_START/END", types[0] == "SESSION_START" and types[-1] == "SESSION_END")
check("đúng 4 PREDICTION", types.count("PREDICTION") == 4)
check("mọi event có ground truth", all(e["session_id"] == sid1 and e["target_word"] == "Năm" and e["target_class"] == nam_class and e["trial"] == 1 and e["timestamp"] for e in events))
preds = [e for e in events if e["event_type"] == "PREDICTION"]
check("prediction_correct == (top1_class == target_class)", all(p["prediction_correct"] == (p["top1_class"] == nam_class) for p in preds))
need = ["top1_class", "top1_label", "top1_confidence", "top2_class", "top2_label", "top2_confidence", "top3_class", "top3_label", "top3_confidence",
        "prediction_correct", "frame_count", "sequence_length", "sequence_complete", "pose_detected", "left_hand_detected", "right_hand_detected",
        "pose_zero_frames", "left_hand_zero_frames", "right_hand_zero_frames", "sequence_id", "sequence_file"]
check("đủ field PREDICTION", all(all(k in p for k in need) for p in preds))
check("frame_count=30 & complete", all(p["frame_count"] == 30 and p["sequence_complete"] for p in preds))
check("phát hiện tay phải = 0 ở lượt 1", preds[0]["right_hand_zero_frames"] == 30 and preds[0]["right_hand_detected"] is False and preds[0]["left_hand_detected"] is True)
check("lượt 2 có tay phải", preds[1]["right_hand_zero_frames"] == 0 and preds[1]["right_hand_detected"] is True)
files_ok = all(os.path.exists(os.path.join(d1, p["sequence_file"])) for p in preds)
check("file sequence tồn tại", files_ok)
arr = np.load(os.path.join(d1, preds[0]["sequence_file"]))
check("npy shape (30,75,3) float32", arr.shape == (30, 75, 3) and arr.dtype == np.float32)
check("không trộn event mode khác (sentence)", len(preds) == 4)
rst = [e for e in events if e["event_type"] == "SEQUENCE_RESET"]
check("SEQUENCE_RESET có frames_before_reset+reason", len(rst) == 1 and rst[0]["frames_before_reset"] == 17 and rst[0]["reason"] == "no_hands_detected")
check("không có trial_result tự động", not any("trial_result" in e for e in events))
sj = json.load(open(os.path.join(d1, "session.json"), encoding="utf-8"))
check("session.json đủ trường", sj["target_word"] == "Năm" and sj["target_class"] == nam_class and sj["trial"] == 1 and sj["mode"] == "words" and sj["start_time"] and sj["end_time"] and sj["checkpoint"].endswith(".pth"), str(sj))

# 6. Trial 2 => thư mục riêng, không overwrite
st, s2 = call("POST", "/api/debug/word/session", {"action": "start", "target_class": nam_class, "trial": 2})
predict("words")
call("POST", "/api/debug/word/session", {"action": "stop"})
check("trial 2 thư mục riêng", s2["session_id"] != sid1 and s2["session_id"].startswith("Nam_trial_02_"), s2["session_id"])
check("log trial 1 không bị ghi đè", len(open(os.path.join(d1, "events.jsonl"), encoding="utf-8").readlines()) == len(events))

# dọn dẹp thư mục test
for d in set(glob.glob(os.path.join(WORDS_DIR, "*"))) - before:
    shutil.rmtree(d, ignore_errors=True)
print("\nKẾT QUẢ:", "TẤT CẢ PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)

