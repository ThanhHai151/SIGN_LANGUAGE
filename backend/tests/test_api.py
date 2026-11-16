"""
Bộ kiểm thử tự động toàn diện cho Hệ thống VSL (VSL Automated Tests)
Kiểm thử:
1. Độ chính xác của bộ nhận diện cử chỉ (Sign Recognizer accuracy & constraints)
2. Độ chính xác của bộ chuyển đổi văn bản sang VSL (NLP Text-to-Sign)
3. Các API endpoints của FastAPI (Health, Recognize, Text-to-Sign, Clear)
"""

import sys
import os
import unittest
import numpy as np

# Thêm đường dẫn gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.models.sign_model import CLASSES
from backend.vision.recognizer import VSLRecognizer
from backend.nlp.text_to_vsl import text_to_vsl_glosses
from fastapi.testclient import TestClient
from backend.app import app

class TestVSLSystem(unittest.TestCase):
    def setUp(self):
        self.recognizer = VSLRecognizer(confidence_threshold=0.7)
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Kiểm tra endpoint /api/health"""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertGreater(data["total_classes"], 20)

    def test_dictionary_endpoint(self):
        """Kiểm tra endpoint /api/dictionary"""
        response = self.client.get("/api/dictionary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("A", data["alphabet"])
        self.assertIn("XIN_CHAO", data["vocabulary"])

    def test_nlp_text_to_sign_greeting(self):
        """Kiểm tra NLP chuyển đổi câu chào và cảm ơn"""
        result = text_to_vsl_glosses("Xin chào, cảm ơn bạn rất nhiều!")
        labels = [r["label"] for r in result]
        self.assertIn("Xin chào", labels)
        self.assertIn("Cảm ơn", labels)
        self.assertIn("Bạn", labels)
        # "rất" là hư từ → bỏ qua, không đánh vần
        self.assertNotIn("Chữ R", labels)

    def test_nlp_fingerspelling_proper_name(self):
        """Kiểm tra NLP tự động chuyển tên riêng chưa có trong từ điển sang đánh vần chữ cái"""
        result = text_to_vsl_glosses("Hải")
        labels = [r["label"] for r in result]
        # H, A, I
        self.assertIn("Chữ H", labels)
        self.assertIn("Chữ A", labels)
        self.assertIn("Chữ I", labels)

    def test_recognize_v_sign(self):
        """Kiểm tra nhận diện cử chỉ chữ V (ngón trỏ và ngón giữa duỗi)"""
        landmarks = [[0.5, 0.7, 0.0] for _ in range(21)]
        landmarks[0] = [0.5, 0.8, 0.0]  # Cổ tay
        landmarks[9] = [0.5, 0.5, 0.0]  # Gốc ngón giữa
        landmarks[8] = [0.42, 0.25, 0.0] # Đầu ngón trỏ (duỗi cao hơn khớp 6)
        landmarks[6] = [0.45, 0.45, 0.0]
        landmarks[12] = [0.58, 0.25, 0.0] # Đầu ngón giữa (duỗi cao hơn khớp 10)
        landmarks[10] = [0.55, 0.45, 0.0]
        landmarks[16] = [0.65, 0.65, 0.0] # Áp út (gập)
        landmarks[14] = [0.65, 0.55, 0.0]
        landmarks[20] = [0.70, 0.70, 0.0] # Út (gập)
        landmarks[18] = [0.70, 0.60, 0.0]
        landmarks[4] = [0.52, 0.65, 0.0]  # Cái (gập)

        pred = self.recognizer.predict_landmarks(landmarks)
        self.assertEqual(pred["sign"], "V")
        self.assertGreaterEqual(pred["confidence"], 0.8)

    def test_recognize_l_sign(self):
        """Kiểm tra nhận diện cử chỉ chữ L (ngón cái và ngón trỏ mở vuông góc)"""
        landmarks = [[0.5, 0.7, 0.0] for _ in range(21)]
        landmarks[0] = [0.5, 0.8, 0.0]  # Cổ tay
        landmarks[9] = [0.5, 0.5, 0.0]  # Gốc ngón giữa
        landmarks[4] = [0.30, 0.50, 0.0] # Ngón cái vươn ngang sang trái
        landmarks[2] = [0.42, 0.65, 0.0]
        landmarks[8] = [0.50, 0.25, 0.0] # Ngón trỏ vươn thẳng đứng lên trên
        landmarks[6] = [0.50, 0.45, 0.0]
        # Các ngón khác gập
        landmarks[12] = [0.55, 0.65, 0.0]
        landmarks[10] = [0.55, 0.55, 0.0]
        landmarks[16] = [0.60, 0.65, 0.0]
        landmarks[14] = [0.60, 0.55, 0.0]
        landmarks[20] = [0.65, 0.65, 0.0]
        landmarks[18] = [0.65, 0.55, 0.0]

        pred = self.recognizer.predict_landmarks(landmarks)
        self.assertEqual(pred["sign"], "L")
        self.assertGreaterEqual(pred["confidence"], 0.85)

    def test_api_text_to_sign_endpoint(self):
        """Kiểm tra endpoint /api/text_to_sign"""
        payload = {"text": "Tôi yêu bạn"}
        response = self.client.post("/api/text_to_sign", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_tokens"], 0)
        seq = data["sequence"]
        # QIPEDC có sẵn MỘT ký hiệu cho cả cụm "tôi yêu bạn" → ghép thành 1 ký hiệu có video mẫu
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq[0]["label"], "Tôi yêu bạn")
        self.assertTrue(seq[0]["video_url"].startswith("https://qipedc.moet.gov.vn/videos/"))

    def test_nlp_two_handed_phrases(self):
        """Kiểm tra dịch thuật các từ vựng 2 tay mới: Xin vui lòng, Giúp đỡ, Gia đình"""
        result = text_to_vsl_glosses("Xin vui lòng giúp đỡ gia đình tôi")
        glosses = [r["gloss"] for r in result]
        self.assertIn("XIN_VUI_LONG", glosses)
        self.assertIn("GIUP_DO", glosses)
        self.assertIn("GIA_DINH", glosses)
        self.assertIn("TOI", glosses)

    def test_two_handed_help_recognition(self):
        """Kiểm tra nhận diện 2 tay cho cử chỉ Giúp đỡ (tay phải nắm, tay trái đỡ dưới)"""
        mock_right = [[0.5, 0.7, 0.0] for _ in range(21)]
        mock_right[0] = [0.5, 0.55, 0.0] # Tay phải ở trên
        mock_right[4] = [0.48, 0.45, 0.0] # Thumbs up

        mock_left = [[0.5, 0.7, 0.0] for _ in range(21)]
        mock_left[0] = [0.5, 0.75, 0.0]  # Tay trái ở dưới làm bệ đỡ
        mock_left[8] = [0.45, 0.55, 0.0]; mock_left[12] = [0.50, 0.53, 0.0]
        mock_left[16] = [0.55, 0.55, 0.0]; mock_left[20] = [0.60, 0.58, 0.0]

        multi_hands = [
            {"handedness": "Right", "landmarks": mock_right},
            {"handedness": "Left", "landmarks": mock_left}
        ]

        response = self.client.post("/api/recognize_landmarks", json={"multi_hands": multi_hands, "landmarks": mock_right})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["sign"], "GIUP_DO")

    def _assert_exact(self, result):
        """Nguyên tắc chính xác: mỗi ký hiệu phải là ký hiệu của ĐÚNG từ được gõ (hoặc cách gọi khác của
        cùng khái niệm trong SAME_SIGN_WORDS) — không dùng ký hiệu của từ gần nghĩa."""
        from backend.nlp.text_to_vsl import headword_keys, SAME_SIGN_WORDS
        from backend.models.lexicon import norm_key
        for r in result:
            if r["type"] != "vocabulary":
                continue
            keys = headword_keys(r["word"])
            ok = r["token"] in keys or norm_key(SAME_SIGN_WORDS.get(r["token"], "")) in keys
            self.assertTrue(ok, f'"{r["token"]}" bị diễn bằng ký hiệu của "{r["word"]}"')

    def test_nlp_everyday_compound_phrases(self):
        """Cụm từ có trong bộ ký hiệu (nấu ăn, lau nhà) được diễn bằng ký hiệu; hư từ bị bỏ; không ghép sai nghĩa"""
        result = text_to_vsl_glosses("Tôi đang học bài và nấu ăn rồi lau nhà sau đó đi ngủ")
        signed = [r["token"] for r in result if r["type"] == "vocabulary" and r.get("motion_id")]
        self.assertIn("nấu ăn", signed)
        self.assertIn("lau nhà", signed)
        self.assertIn("học", signed)
        tokens = [r["token"] for r in result]
        self.assertNotIn("và", tokens)   # hư từ "và", "rồi", "đang" bị bỏ qua
        self.assertNotIn("rồi", tokens)
        self._assert_exact(result)

    def test_nlp_full_long_discourse_sentence(self):
        """Đoạn văn dài: từ có ký hiệu được diễn bằng ký hiệu của chính từ đó, từ chưa có ký hiệu được đánh vần cả từ"""
        text = (
            "Những ngày cuối tuần của tôi không có lịch trình cụ thể vì chúng phụ thuộc vào mức độ bận rộn của tôi trong tuần. "
            "Tôi thường dành một ít thời gian cho việc học tập hoặc các dự án cá nhân, nhưng tôi cũng đảm bảo mình có đủ thời gian để thư giãn. "
            "Nếu có thể, tôi thích gặp gỡ bạn bè hoặc khám phá những địa điểm mới, vì điều đó cho tôi cơ hội nghỉ ngơi khỏi thói quen thường ngày."
        )
        result = text_to_vsl_glosses(text)
        self.assertGreater(len(result), 15)
        self._assert_exact(result)
        signed = [r["token"] for r in result if r["type"] == "vocabulary" and r.get("motion_id")]
        for w in ("cuối tuần", "không có", "phụ thuộc", "học tập", "dự án", "thư giãn", "gặp gỡ", "khám phá",
                  "địa điểm", "mới", "cơ hội", "thói quen"):
            self.assertIn(w, signed)
        # Từ ghép chưa có ký hiệu ("lịch trình", "bạn bè") được đánh vần CẢ TỪ, không tách ra từng tiếng để diễn sai nghĩa
        spelled_words = {r.get("spelled_word") for r in result if r["type"] == "fingerspelling"}
        self.assertIn("lịch trình", spelled_words)
        self.assertIn("bạn bè", spelled_words)
        # Mọi ký hiệu được diễn đều có video mẫu người thật (miền Nam / dùng chung)
        self.assertTrue(all(r.get("video_url") for r in result if r.get("motion_id")))

    def test_nlp_compound_not_split_into_wrong_signs(self):
        """"sinh nhật", "công nhân", "kết hôn": không được diễn "nhật" (nước Nhật), "nhân" (phép nhân), "hôn" """
        for text, bad in (("Chúc mừng sinh nhật", "nhật"), ("Chú tôi là công nhân", "nhân"), ("Bố mẹ kết hôn", "hôn"), ("Không sao", "sao")):
            result = text_to_vsl_glosses(text)
            self.assertNotIn(bad, [r["token"] for r in result if r["type"] == "vocabulary"], text)
            self._assert_exact(result)

    # ------------------------------------------------------------------
    # Bộ từ vựng thống nhất (QIPEDC + giáo trình 102 + robot)
    # ------------------------------------------------------------------
    def test_lexicon_endpoint(self):
        """/api/lexicon trả về bộ từ vựng thống nhất với hàng nghìn mục từ"""
        data = self.client.get("/api/lexicon").json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total"], 3000)
        common = self.client.get("/api/lexicon", params={"filter": "common"}).json()
        self.assertGreater(common["total"], 500)
        self.assertTrue(all(e["common"] for e in common["entries"]))

    def test_lexicon_search(self):
        """Tìm kiếm theo từ khóa (có dấu)"""
        data = self.client.get("/api/lexicon", params={"q": "gia đình"}).json()
        words = [e["word"] for e in data["entries"]]
        self.assertIn("gia đình", words)

    def test_camera_classes_linked_to_lexicon(self):
        """Cả 102 lớp của mô hình camera đều trỏ tới một mục từ trong bộ từ vựng chung"""
        cam = self.client.get("/api/lexicon", params={"filter": "camera"}).json()
        self.assertEqual(cam["total"], 102)
        self.assertEqual(sorted(e["sstcn_index"] for e in cam["entries"]), list(range(102)))
        classes = self.client.get("/api/vsl_102_classes").json()
        self.assertTrue(all(s.get("entry_id") for s in classes["signs"]))

    def test_camera_and_text_share_vocabulary(self):
        """Từ nhận diện được bằng camera cũng dịch ngược được từ văn bản về CÙNG mục từ"""
        from backend.models.lexicon import get_entry_by_sstcn
        for idx, word in [(51, "tôi"), (52, "chào"), (60, "bạn"), (75, "ăn"), (80, "gia đình"), (84, "mẹ"), (98, "bác sĩ")]:
            entry = get_entry_by_sstcn(idx)
            seq = text_to_vsl_glosses(word)
            self.assertEqual(seq[0]["entry_id"], entry["id"], entry["word"])

    def test_south_standard(self):
        """Chuẩn miền Nam: chỉ còn video miền Nam / dùng chung; từ chỉ có video miền Bắc/Trung → không có video"""
        data = self.client.get("/api/lexicon", params={"q": "địa chỉ"}).json()
        entry = next(e for e in data["entries"] if e["word"] == "địa chỉ")
        self.assertEqual({v["region"] for v in entry["variants"]} - {"GT"}, {"N"})
        self.assertTrue(entry["video_url"].endswith("N.mp4"))
        self.assertEqual(entry["region"], "N")
        lex = self.client.get("/api/lexicon").json()["entries"]
        for e in lex:
            for v in e.get("variants", []):
                self.assertNotIn(v["region"], ("B", "T"), e["word"])
        # "bạn": robot / video dùng ký hiệu miền Nam
        seq = self.client.post("/api/text_to_sign", json={"text": "bạn"}).json()["sequence"]
        self.assertTrue(seq[0]["video_id"].endswith("N"))
        # từ không có video miền Nam: không dùng tư thế vẽ tay, robot đánh vần
        no_south = next(e for e in lex if e.get("no_south") and e.get("source") == "QIPEDC")
        self.assertIsNone(no_south["video_url"])
        item = self.client.post("/api/text_to_sign", json={"text": no_south["keys"][0]}).json()["sequence"][0]
        self.assertFalse(item["has_pose"])
        self.assertTrue(item["spell"])

    # ------------------------------------------------------------------
    # Chuyển động robot học từ video QIPEDC
    # ------------------------------------------------------------------
    def test_motion_endpoint(self):
        """/api/motion trả về keyframes góc khớp robot học từ video người thật"""
        from backend.models.lexicon import motion_ids
        ids = sorted(motion_ids())
        self.assertGreater(len(ids), 50)
        data = self.client.get(f"/api/motion/{ids[0]}").json()
        m = data["motion"]
        self.assertGreater(len(m["keyframes"]), 5)
        kf = m["keyframes"][len(m["keyframes"]) // 2]
        for key in ["rightArm", "rightForearm", "rightTwist", "rightWrist", "rightFingers", "leftArm", "leftFingers"]:
            self.assertIn(key, kf)
        self.assertEqual(set(kf["rightFingers"]), {"thumb", "index", "middle", "ring", "pinky"})
        a0, a1 = m["active"]
        self.assertTrue(0 <= a0 < a1 <= m["duration_ms"] / 1000 + 0.5)
        self.assertEqual(self.client.get("/api/motion/khong-co").status_code, 404)
        self.assertEqual(self.client.get("/api/motion/..%2Fapp").status_code, 404)

    def test_text_to_sign_uses_learned_motion(self):
        """Câu nhập vào → ký hiệu có chuyển động học từ video được robot diễn (motion_id + đoạn có động tác)"""
        seq = text_to_vsl_glosses("Xin chào bạn, xin lỗi")
        learned = [s for s in seq if s.get("motion_id")]
        self.assertGreaterEqual(len(learned), 2)
        for s in learned:
            self.assertTrue(s["has_pose"])
            self.assertEqual(len(s["clip"]), 2)
            self.assertIn(s["motion_id"], s["video_url"])

    def test_motion_files_layout(self):
        """Mỗi khung hình chuyển động có đúng số kênh góc khớp theo LAYOUT của robot"""
        import json, glob
        from backend.training.retarget_motion import LAYOUT
        from backend.models.lexicon import MOTION_DIR
        files = sorted(glob.glob(os.path.join(MOTION_DIR, "[DW]*.json")))[:5]
        self.assertTrue(files)
        for f in files:
            m = json.load(open(f, encoding="utf-8"))
            self.assertEqual(len(m["frames"][0]), len(LAYOUT))

    def test_sign_description(self):
        """Mỗi ký hiệu robot đã học có phần diễn tả cách làm: số tay, hình dạng, lòng bàn tay, vị trí, chuyển động"""
        seq = text_to_vsl_glosses("Xin chào bạn, xin lỗi")
        described = [s for s in seq if s.get("sign_description")]
        self.assertGreaterEqual(len(described), 2)
        for s in described:
            d = s["sign_description"]
            self.assertTrue(d["summary"])
            self.assertIn(d["hands"].split(" ")[0], ("Hai", "Một"))
            for p in d["parts"]:
                for k in ("side", "shape", "palm", "point", "location", "movement"):
                    self.assertTrue(p[k], k)
        r = self.client.get("/api/lexicon/" + described[0]["entry_id"])
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["entry"]["sign_description"])

    def test_media_cache_serves_ranges(self):
        """Video mẫu phát qua bộ đệm backend: hỗ trợ tua (HTTP Range), cache dài hạn, không nén gzip"""
        from backend.models import video_cache
        path = video_cache.cache_path("videos", "ZZTEST")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(bytes(range(256)) * 8)
        try:
            r = self.client.get("/api/media/videos/ZZTEST.mp4", headers={"Range": "bytes=10-19", "Accept-Encoding": "gzip"})
            self.assertEqual(r.status_code, 206)
            self.assertEqual(r.content, bytes(range(10, 20)))
            self.assertIn("immutable", r.headers["cache-control"])
            self.assertNotEqual(r.headers.get("content-encoding"), "gzip")
            self.assertEqual(self.client.get("/api/media/other/ZZTEST.mp4").status_code, 404)
        finally:
            os.remove(path)

    def test_learn_new_motion(self):
        """Từ chưa có chuyển động: nhận điểm khớp từ trình duyệt → lưu chuyển động robot + diễn tả (không cần scipy)"""
        import json, shutil
        from backend.models.lexicon import MOTION_DIR, DESC_PATH
        raw = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "raw_W03990.json"), encoding="utf-8"))
        bak = DESC_PATH + ".bak"
        shutil.copy(DESC_PATH, bak)
        try:
            payload = {"id": "ZZLEARN", "fps": raw["fps"], "dur": raw["dur"],
                       "frames": [{k: f.get(k) for k in ("p", "l", "r")} for f in raw["frames"]]}
            r = self.client.post("/api/motion/learn", json=payload).json()
            self.assertTrue(r["success"])
            self.assertEqual(len(r["clip"]), 2)
            self.assertEqual(self.client.get("/api/motion/ZZLEARN").status_code, 200)
            bad = self.client.post("/api/motion/learn", json={"id": "../x", "frames": []}).json()
            self.assertFalse(bad["success"])
        finally:
            from backend.models.motion_learn import RAW_DIR
            for p in (os.path.join(MOTION_DIR, "ZZLEARN.json"), os.path.join(RAW_DIR, "ZZLEARN.json")):
                if os.path.exists(p):
                    os.remove(p)
            shutil.move(bak, DESC_PATH)

if __name__ == "__main__":
    unittest.main()

