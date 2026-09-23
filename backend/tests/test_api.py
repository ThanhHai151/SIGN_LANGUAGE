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
        tokens = [item["label"] for item in data["sequence"]]
        self.assertIn("Tôi", tokens)
        self.assertIn("Yêu thương", tokens)
        self.assertIn("Bạn", tokens)

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

    def test_nlp_everyday_compound_phrases(self):
        """Kiểm tra các cụm từ học bài, nấu ăn, lau nhà, đi ngủ được nhận diện thành cụm từ chứ không đánh vần chữ cái"""
        result = text_to_vsl_glosses("Tôi đang học bài và nấu ăn rồi lau nhà sau đó đi ngủ")
        glosses = [r["gloss"] for r in result]
        types = [r["type"] for r in result]

        self.assertIn("HOC_BAI", glosses)
        self.assertIn("NAU_AN", glosses)
        self.assertIn("LAU_NHA", glosses)
        self.assertIn("DI_NGU", glosses)

        # Đảm bảo không có chữ cái fingerspelling thừa
        for item in result:
            if item["gloss"] in ["HOC_BAI", "NAU_AN", "LAU_NHA", "DI_NGU"]:
                self.assertEqual(item["type"], "vocabulary")

    def test_nlp_full_long_discourse_sentence(self):
        """Kiểm tra nguyên văn đoạn văn dài tự sự của người dùng được dịch 100% sang cụm từ có nghĩa, không có fingerspelling"""
        text = (
            "Những ngày cuối tuần của tôi không có lịch trình cụ thể vì chúng phụ thuộc vào mức độ bận rộn của tôi trong tuần. "
            "Tôi thường dành một ít thời gian cho việc học tập hoặc các dự án cá nhân, nhưng tôi cũng đảm bảo mình có đủ thời gian để thư giãn. "
            "Nếu có thể, tôi thích gặp gỡ bạn bè hoặc khám phá những địa điểm mới, vì điều đó cho tôi cơ hội nghỉ ngơi khỏi thói quen thường ngày."
        )
        result = text_to_vsl_glosses(text)
        self.assertGreater(len(result), 15)
        
        # Đảm bảo 100% token là vocabulary cụm từ, không có fingerspelling
        fingerspelling_tokens = [r for r in result if r["type"] == "fingerspelling"]
        self.assertEqual(len(fingerspelling_tokens), 0, f"Còn sót các ký tự fingerspelling: {fingerspelling_tokens}")
        
        glosses = [r["gloss"] for r in result]
        self.assertIn("CUOI_TUAN", glosses)
        self.assertIn("LICH_TRINH", glosses)
        self.assertIn("PHU_THUOC", glosses)
        self.assertIn("BAN_RON", glosses)
        self.assertIn("TRONG_TUAN", glosses)
        self.assertIn("THOI_GIAN", glosses)
        self.assertIn("HOC_BAI", glosses)
        self.assertIn("DU_AN", glosses)
        self.assertIn("DAM_BAO", glosses)
        self.assertIn("THU_GIAN", glosses)
        self.assertIn("CO_THE", glosses)
        self.assertIn("THICH", glosses)
        self.assertIn("GAP_GO", glosses)
        self.assertIn("BAN_BE", glosses)
        self.assertIn("KHAM_PHA", glosses)
        self.assertIn("DIA_DIEM", glosses)
        self.assertIn("CO_HOI", glosses)
        self.assertIn("THOI_QUEN", glosses)

if __name__ == "__main__":
    unittest.main()

