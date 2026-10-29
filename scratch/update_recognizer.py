# -*- coding: utf-8 -*-
import sys
import re

path = 'backend/vision/recognizer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Thêm import và khởi tạo digit_10_detector
old_init = """        self.alphabet_classifier = alphabet_classifier
        from backend.vision.digit_aggregator import VSLSequentialDigitAggregator
        self.digit_aggregator = VSLSequentialDigitAggregator()"""

new_init = """        self.alphabet_classifier = alphabet_classifier
        from backend.vision.digit_aggregator import VSLSequentialDigitAggregator
        self.digit_aggregator = VSLSequentialDigitAggregator()
        from backend.vision.digit_10_detector import VSL10GestureDetector
        self.digit_10_detector = VSL10GestureDetector()"""

assert old_init in content, "old_init not found"
content = content.replace(old_init, new_init, 1)

# 2. Thêm reset detector vào đổi session/mode
old_mode_reset = """            if hasattr(self, "digit_aggregator"):
                self.digit_aggregator.reset()"""

new_mode_reset = """            if hasattr(self, "digit_aggregator"):
                self.digit_aggregator.reset()
            if hasattr(self, "digit_10_detector"):
                self.digit_10_detector.reset()"""

assert old_mode_reset in content, "old_mode_reset not found"
content = content.replace(old_mode_reset, new_mode_reset, 1)

# 3. Thêm reset detector vào trường hợp not p_prim
old_no_hand = """        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
            if is_digits_mode and hasattr(self, "digit_aggregator"):
                agg_out = self.digit_aggregator.update("NEUTRAL", 0.0, timestamp=current_time)"""

new_no_hand = """        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
            if is_digits_mode:
                if hasattr(self, "digit_10_detector"):
                    self.digit_10_detector.reset()
                if hasattr(self, "digit_aggregator"):
                    agg_out = self.digit_aggregator.update("NEUTRAL", 0.0, timestamp=current_time)"""

assert old_no_hand in content, "old_no_hand not found"
content = content.replace(old_no_hand, new_no_hand, 1)

# 4. Cập nhật nhánh is_digits_mode
old_digits_block = """        # -------------------------------------------------------------
        # XỬ LÝ CHUYÊN BIỆT CHO CHẾ ĐỘ CHỮ SỐ (digits mode)
        # -------------------------------------------------------------
        if is_digits_mode:
            from backend.vision.digit_classifier import digit_classifier
            digit_res = digit_classifier.classify_detailed(p_prim)
            if digit_res["status"] == "MATCH" and digit_res["digit"] in DIGITS_MODE_SIGNS:
                raw_digit = digit_res["digit"]
                raw_conf = digit_res["confidence"]
            else:
                raw_digit = "NEUTRAL"
                raw_conf = 0.0

            # Chặn tuyệt đối: chế độ digits không bao giờ nhận diện chữ cái hoặc từ vựng
            if raw_digit not in DIGITS_MODE_SIGNS:
                raw_digit = "NEUTRAL"
                raw_conf = 0.0

            self.prediction_history.append((raw_digit, raw_conf))

            # Smoothing cho chữ số đơn tức thời
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            confirmed = False
            new_token_added = None

            if len(recent_signs) >= self.smoothing_window // 2:
                smoothed_digit, count = collections.Counter(recent_signs).most_common(1)[0]
                smoothed_conf = raw_conf if count >= (self.smoothing_window // 2) else 0.0
                if count >= (self.smoothing_window // 2) and smoothed_digit in DIGITS_MODE_SIGNS:
                    if smoothed_digit != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                        self.last_confirmed_sign = smoothed_digit
                        self.last_confirmed_time = current_time
                        confirmed = True
                        label_text = f"Số {smoothed_digit}"
                        self.sentence_tokens.append(label_text)
                        new_token_added = label_text
            else:
                smoothed_digit = raw_digit
                smoothed_conf = raw_conf

            # Cập nhật bộ ghép số tuần tự (Sequential Digit Aggregator)
            agg_out = self.digit_aggregator.update(
                sign=smoothed_digit if smoothed_conf >= self.confidence_threshold else "NEUTRAL",
                confidence=smoothed_conf,
                timestamp=current_time
            )

            # Khi aggregator chốt số nhiều chữ số, cập nhật thay thế token đơn lẻ phía trước trong câu
            if agg_out.is_committed and agg_out.committed_number:
                committed_str = agg_out.committed_number
                multi_label = f"Số {committed_str}"
                if len(committed_str) > 1:
                    prefix_label = f"Số {committed_str[0]}"
                    if self.sentence_tokens and self.sentence_tokens[-1] == prefix_label:
                        self.sentence_tokens[-1] = multi_label
                    else:
                        self.sentence_tokens.append(multi_label)
                    confirmed = True
                    new_token_added = multi_label
                    self.last_confirmed_sign = committed_str
                    self.last_confirmed_time = current_time

            # Xác định nhãn hiển thị và giá trị ký hiệu:
            # Ưu tiên số đã chốt -> số tạm thời đang gom -> chữ số tức thời -> NEUTRAL
            if agg_out.is_committed and agg_out.committed_number:
                display_sign = agg_out.committed_number
            elif agg_out.provisional_number is not None:
                display_sign = agg_out.provisional_number
            elif raw_digit in DIGITS_MODE_SIGNS:
                display_sign = raw_digit
            else:
                display_sign = "NEUTRAL"

            is_valid_digit_sign = (display_sign != "NEUTRAL" and display_sign.isdigit())
            display_label = f"Số {display_sign}" if is_valid_digit_sign else "Đang theo dõi cử chỉ..."

            self._log_diagnostic(
                p_prim=p_prim,
                handedness=handedness,
                finger_states=None,
                geom_sign=digit_res.get("digit", "NEUTRAL"),
                geom_conf=digit_res.get("confidence", 0.0),
                has_mlp=False,
                nn_sign="NEUTRAL",
                nn_conf=0.0,
                final_sign=display_sign,
                final_conf=raw_conf,
                confirmed=confirmed,
                expected_sign=expected_sign,
                session_id=session_id,
                mode=mode,
                classifier_details=digit_res
            )

            return {
                "sign": display_sign,
                "label": display_label,
                "type": "digit" if is_valid_digit_sign else "unknown",
                "confidence": round(float(raw_conf), 2),
                "confirmed": confirmed,
                "new_token": new_token_added,
                "sentence": " ".join(self.sentence_tokens),
                "description": f"Chữ số {display_sign}" if is_valid_digit_sign else "",
                "aggregator": agg_out.to_dict()
            }"""

new_digits_block = """        # -------------------------------------------------------------
        # XỬ LÝ CHUYÊN BIỆT CHO CHẾ ĐỘ CHỮ SỐ (digits mode)
        # -------------------------------------------------------------
        if is_digits_mode:
            # 1. ƯU TIÊN 1: Nhận diện cử chỉ động riêng cho Số 10 VSL
            det10_res = self.digit_10_detector.update(p_prim, timestamp=current_time)
            if det10_res["detected"]:
                # Chốt trực tiếp Số 10
                self.digit_aggregator.reset()
                self.prediction_history.clear()
                
                label_text = "Số 10"
                # Nếu frame trước đó lỡ chốt nhầm Số 1 do smoothing tức thời của cùng cử chỉ này, thay thế nó
                if self.sentence_tokens and self.sentence_tokens[-1] == "Số 1":
                    self.sentence_tokens[-1] = label_text
                elif not (self.sentence_tokens and self.sentence_tokens[-1] == label_text):
                    self.sentence_tokens.append(label_text)
                    
                self.last_confirmed_sign = "10"
                self.last_confirmed_time = current_time
                
                self._log_diagnostic(
                    p_prim=p_prim,
                    handedness=handedness,
                    finger_states=None,
                    geom_sign="10",
                    geom_conf=det10_res["confidence"],
                    has_mlp=False,
                    nn_sign="NEUTRAL",
                    nn_conf=0.0,
                    final_sign="10",
                    final_conf=det10_res["confidence"],
                    confirmed=True,
                    expected_sign=expected_sign,
                    session_id=session_id,
                    mode=mode,
                    classifier_details={"detector_10": det10_res}
                )
                
                return {
                    "sign": "10",
                    "label": label_text,
                    "type": "digit",
                    "confidence": round(float(det10_res["confidence"]), 2),
                    "confirmed": True,
                    "new_token": label_text,
                    "sentence": " ".join(self.sentence_tokens),
                    "description": "Chữ số 10",
                    "aggregator": self.digit_aggregator.to_dict(),
                    "detector_10": det10_res
                }

            # Kiểm tra thời gian cooldown sau khi vừa chốt Số 10
            is_in_10_cooldown = (current_time - self.digit_10_detector.last_detected_time < self.digit_10_detector.config.cooldown_sec)

            # 2. Phân loại hình học đơn frame cho chữ số 0-9
            from backend.vision.digit_classifier import digit_classifier
            digit_res = digit_classifier.classify_detailed(p_prim)
            if (not is_in_10_cooldown) and digit_res["status"] == "MATCH" and digit_res["digit"] in DIGITS_MODE_SIGNS:
                raw_digit = digit_res["digit"]
                raw_conf = digit_res["confidence"]
            else:
                raw_digit = "NEUTRAL"
                raw_conf = 0.0

            # Chặn tuyệt đối: chế độ digits không bao giờ nhận diện chữ cái hoặc từ vựng
            if raw_digit not in DIGITS_MODE_SIGNS:
                raw_digit = "NEUTRAL"
                raw_conf = 0.0

            self.prediction_history.append((raw_digit, raw_conf))

            # Smoothing cho chữ số đơn tức thời
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            confirmed = False
            new_token_added = None

            # Hoãn chốt số 1 nếu detector đang quan sát chuyển động xoay/nghiêng của Số 10
            is_observing_10 = det10_res.get("is_observing", False)

            if len(recent_signs) >= self.smoothing_window // 2:
                smoothed_digit, count = collections.Counter(recent_signs).most_common(1)[0]
                smoothed_conf = raw_conf if count >= (self.smoothing_window // 2) else 0.0
                if count >= (self.smoothing_window // 2) and smoothed_digit in DIGITS_MODE_SIGNS:
                    should_defer_one = (smoothed_digit == "1" and is_observing_10)
                    if not should_defer_one:
                        if smoothed_digit != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                            self.last_confirmed_sign = smoothed_digit
                            self.last_confirmed_time = current_time
                            confirmed = True
                            label_text = f"Số {smoothed_digit}"
                            self.sentence_tokens.append(label_text)
                            new_token_added = label_text
            else:
                smoothed_digit = raw_digit
                smoothed_conf = raw_conf

            # 3. ƯU TIÊN 2: Bộ ghép số tuần tự (Sequential Digit Aggregator)
            feed_digit = smoothed_digit
            if (feed_digit == "1" and is_observing_10) or is_in_10_cooldown:
                feed_digit = "NEUTRAL"
                feed_conf = 0.0
            else:
                feed_conf = smoothed_conf if smoothed_conf >= self.confidence_threshold else 0.0

            agg_out = self.digit_aggregator.update(
                sign=feed_digit,
                confidence=feed_conf,
                timestamp=current_time
            )

            # Khi aggregator chốt số nhiều chữ số, cập nhật thay thế token đơn lẻ phía trước trong câu
            if agg_out.is_committed and agg_out.committed_number:
                committed_str = agg_out.committed_number
                multi_label = f"Số {committed_str}"
                if len(committed_str) > 1:
                    prefix_label = f"Số {committed_str[0]}"
                    if self.sentence_tokens and self.sentence_tokens[-1] == prefix_label:
                        self.sentence_tokens[-1] = multi_label
                    else:
                        self.sentence_tokens.append(multi_label)
                    confirmed = True
                    new_token_added = multi_label
                    self.last_confirmed_sign = committed_str
                    self.last_confirmed_time = current_time

            # 4. ƯU TIÊN 3: Xác định nhãn hiển thị và giá trị ký hiệu đơn 0-9
            if agg_out.is_committed and agg_out.committed_number:
                display_sign = agg_out.committed_number
            elif agg_out.provisional_number is not None:
                display_sign = agg_out.provisional_number
            elif raw_digit in DIGITS_MODE_SIGNS:
                display_sign = raw_digit
            else:
                display_sign = "NEUTRAL"

            is_valid_digit_sign = (display_sign != "NEUTRAL" and display_sign.isdigit())
            display_label = f"Số {display_sign}" if is_valid_digit_sign else "Đang theo dõi cử chỉ..."

            self._log_diagnostic(
                p_prim=p_prim,
                handedness=handedness,
                finger_states=None,
                geom_sign=digit_res.get("digit", "NEUTRAL"),
                geom_conf=digit_res.get("confidence", 0.0),
                has_mlp=False,
                nn_sign="NEUTRAL",
                nn_conf=0.0,
                final_sign=display_sign,
                final_conf=raw_conf,
                confirmed=confirmed,
                expected_sign=expected_sign,
                session_id=session_id,
                mode=mode,
                classifier_details=digit_res
            )

            return {
                "sign": display_sign,
                "label": display_label,
                "type": "digit" if is_valid_digit_sign else "unknown",
                "confidence": round(float(raw_conf), 2),
                "confirmed": confirmed,
                "new_token": new_token_added,
                "sentence": " ".join(self.sentence_tokens),
                "description": f"Chữ số {display_sign}" if is_valid_digit_sign else "",
                "aggregator": agg_out.to_dict()
            }"""

assert old_digits_block in content, "old_digits_block not found"
content = content.replace(old_digits_block, new_digits_block, 1)

# 5. Cập nhật clear_sentence và thêm reset()
old_clear = """    def clear_sentence(self):
        \"\"\"Xóa trắng câu hiện tại\"\"\"
        self.sentence_tokens = []
        self.last_confirmed_sign = None
        return {"sentence": ""}"""

new_clear = """    def clear_sentence(self):
        \"\"\"Xóa trắng câu hiện tại\"\"\"
        self.sentence_tokens = []
        self.last_confirmed_sign = None
        if hasattr(self, "digit_aggregator"):
            self.digit_aggregator.reset()
        if hasattr(self, "digit_10_detector"):
            self.digit_10_detector.reset()
        return {"sentence": ""}

    def reset(self):
        \"\"\"Reset toàn bộ trạng thái bộ nhận diện\"\"\"
        self.prediction_history.clear()
        self.last_confirmed_sign = None
        self.last_confirmed_time = 0
        self.sentence_tokens = []
        if hasattr(self, "digit_aggregator"):
            self.digit_aggregator.reset()
        if hasattr(self, "digit_10_detector"):
            self.digit_10_detector.reset()"""

assert old_clear in content, "old_clear not found"
content = content.replace(old_clear, new_clear, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated backend/vision/recognizer.py successfully.")

