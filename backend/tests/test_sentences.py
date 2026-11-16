"""
Kiểm thử chuyển văn bản → ký hiệu trên bộ 90 câu mẫu (trường học, giao tiếp cơ bản, gia đình).
Chạy: python -m unittest backend.tests.test_sentences
Xem chi tiết từng câu: python -m backend.training.eval_sentences --detail
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.training.eval_sentences import evaluate, TOPICS

# Mức tối thiểu hiện đạt được; chỉ được tăng khi bổ sung ký hiệu, không được tụt.
MIN_WORD_RATE = 0.78
MIN_SENTENCES_OK = 40


class TestSampleSentences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = evaluate()

    def test_fixture_size(self):
        """Bộ câu mẫu có 90 câu, mỗi chủ đề 30 câu."""
        self.assertEqual(self.res["total"]["sentences"], 90)
        for key in TOPICS:
            self.assertEqual(self.res["by_topic"][key]["sentences"], 30)

    def test_every_sentence_produces_signs(self):
        """Câu nào cũng cho ra chuỗi ký hiệu và có ít nhất một từ robot diễn bằng ký hiệu."""
        for r in self.res["rows"]:
            self.assertTrue(r["glosses"], r["text"])
            self.assertGreater(r["n_words_signed"], 0, r["text"])

    def test_required_words_rate(self):
        """Tỉ lệ từ bắt buộc được diễn bằng ký hiệu không tụt dưới mức đã đạt."""
        t = self.res["total"]
        rate = t["must_ok"] / t["must"]
        self.assertGreaterEqual(rate, MIN_WORD_RATE, f"{t['must_ok']}/{t['must']}: {dict(self.res['missing_words'])}")
        self.assertGreaterEqual(t["ok"], MIN_SENTENCES_OK)

    def test_core_words_are_signed(self):
        """Các từ cốt lõi không bao giờ được đánh vần."""
        core = {"tôi", "bạn", "xin chào", "cảm ơn", "xin lỗi", "bố", "mẹ", "gia đình", "trường", "học sinh", "cô giáo"}
        self.assertFalse(core & set(self.res["missing_words"]), self.res["missing_words"])


if __name__ == "__main__":
    unittest.main()
