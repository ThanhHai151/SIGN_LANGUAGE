import sys

with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add self.digit_aggregator initialization
target_init = "        self.alphabet_classifier = alphabet_classifier"
replacement_init = """        self.alphabet_classifier = alphabet_classifier
        from backend.vision.digit_aggregator import VSLSequentialDigitAggregator
        self.digit_aggregator = VSLSequentialDigitAggregator()"""

assert target_init in content, "target_init not found"
content = content.replace(target_init, replacement_init, 1)

# 2. Add timestamp parameter to predict_landmarks signature
target_sig = "def predict_landmarks(self, landmarks_list, landmarks_secondary=None, multi_hands=None, expected_sign=None, session_id=None, mode=None):"
replacement_sig = "def predict_landmarks(self, landmarks_list, landmarks_secondary=None, multi_hands=None, expected_sign=None, session_id=None, mode=None, timestamp=None):"

assert target_sig in content, "target_sig not found"
content = content.replace(target_sig, replacement_sig, 1)

# 3. Add current_time and aggregator reset on mode switch
target_mode_switch = """        if self._session_last_modes.get(session_key) != mode:
            self.prediction_history.clear()
            self.last_confirmed_sign = None
            self._session_last_modes[session_key] = mode"""

replacement_mode_switch = """        current_time = time.time() if timestamp is None else float(timestamp)

        if self._session_last_modes.get(session_key) != mode:
            self.prediction_history.clear()
            self.last_confirmed_sign = None
            self._session_last_modes[session_key] = mode
            if hasattr(self, "digit_aggregator"):
                self.digit_aggregator.reset()"""

assert target_mode_switch in content, "target_mode_switch not found"
content = content.replace(target_mode_switch, replacement_mode_switch, 1)

# 4. Handle p_prim empty/invalid when is_digits_mode
target_empty_prim = """        if not p_prim or len(p_prim) < 21:
            return {
                "sign": "NEUTRAL",
                "label": "Chờ cử chỉ...",
                "type": "unknown",
                "confidence": 0.0,
                "confirmed": False,
                "new_token": None,
                "sentence": " ".join(self.sentence_tokens)
            }"""

replacement_empty_prim = """        if not p_prim or len(p_prim) < 21:
            if is_digits_mode and hasattr(self, "digit_aggregator"):
                agg_out = self.digit_aggregator.update("NEUTRAL", 0.0, timestamp=current_time)
                if agg_out.is_committed and agg_out.committed_number:
                    label_text = f"Số {agg_out.committed_number}"
                    self.sentence_tokens.append(label_text)
                    self.last_confirmed_sign = agg_out.committed_number
                    self.last_confirmed_time = current_time
                    return {
                        "sign": agg_out.committed_number,
                        "label": label_text,
                        "type": "digit",
                        "confidence": 1.0,
                        "confirmed": True,
                        "new_token": label_text,
                        "sentence": " ".join(self.sentence_tokens),
                        "description": f"Chữ số {agg_out.committed_number}",
                        "aggregator": agg_out.to_dict()
                    }
            return {
                "sign": "NEUTRAL",
                "label": "Chờ cử chỉ...",
                "type": "unknown",
                "confidence": 0.0,
                "confirmed": False,
                "new_token": None,
                "sentence": " ".join(self.sentence_tokens)
            }"""

assert target_empty_prim in content, "target_empty_prim not found"
content = content.replace(target_empty_prim, replacement_empty_prim, 1)

# 5. Update is_digits_mode block
target_digits_block = """        # -------------------------------------------------------------
        # XỬ LÝ CHUYÊN BIỆT CHO CHẾ ĐỘ CHỮ SỐ (digits mode)
        # -------------------------------------------------------------
        if is_digits_mode:
            from backend.vision.digit_classifier import digit_classifier
            digit_res = digit_classifier.classify_detailed(p_prim)
            if digit_res["status"] == "MATCH" and digit_res["digit"] in DIGITS_MODE_SIGNS:
                final_sign = digit_res["digit"]
                final_conf = digit_res["confidence"]
            else:
                final_sign = "NEUTRAL"
                final_conf = 0.0

            # Chặn tuyệt đối: chế độ digits không bao giờ trả về chữ cái hoặc từ vựng
            if final_sign not in DIGITS_MODE_SIGNS:
                final_sign = "NEUTRAL"
                final_conf = 0.0

            self.prediction_history.append((final_sign, final_conf))

            # Smoothing cho chữ số
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            confirmed = False
            new_token_added = None

            if len(recent_signs) >= self.smoothing_window // 2:
                most_common, count = collections.Counter(recent_signs).most_common(1)[0]
                if count >= (self.smoothing_window // 2) and most_common in DIGITS_MODE_SIGNS:
                    current_time = time.time()
                    if most_common != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                        self.last_confirmed_sign = most_common
                        self.last_confirmed_time = current_time
                        confirmed = True
                        label_text = f"Số {most_common}"
                        self.sentence_tokens.append(label_text)
                        new_token_added = label_text

            sign_meta = ALL_SIGNS.get(final_sign, {})
            display_label = f"Số {final_sign}" if final_sign in DIGITS_MODE_SIGNS else "Đang theo dõi cử chỉ..."

            self._log_diagnostic(
                p_prim=p_prim,
                handedness=handedness,
                finger_states=None,
                geom_sign=digit_res.get("digit", "NEUTRAL"),
                geom_conf=digit_res.get("confidence", 0.0),
                has_mlp=False,
                nn_sign="NEUTRAL",
                nn_conf=0.0,
                final_sign=final_sign,
                final_conf=final_conf,
                confirmed=confirmed,
                expected_sign=expected_sign,
                session_id=session_id,
                mode=mode,
                classifier_details=digit_res
            )

            return {
                "sign": final_sign,
                "label": display_label,
                "type": "digit" if final_sign in DIGITS_MODE_SIGNS else "unknown",
                "confidence": round(float(final_conf), 2),
                "confirmed": confirmed,
                "new_token": new_token_added,
                "sentence": " ".join(self.sentence_tokens),
                "description": f"Chữ số {final_sign}" if final_sign in DIGITS_MODE_SIGNS else ""
            }"""

replacement_digits_block = """        # -------------------------------------------------------------
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
            if len(recent_signs) >= self.smoothing_window // 2:
                smoothed_digit, count = collections.Counter(recent_signs).most_common(1)[0]
                smoothed_conf = raw_conf if count >= (self.smoothing_window // 2) else 0.0
            else:
                smoothed_digit = raw_digit
                smoothed_conf = raw_conf

            # Cập nhật bộ ghép số tuần tự (Sequential Digit Aggregator)
            agg_out = self.digit_aggregator.update(
                sign=smoothed_digit if smoothed_conf >= self.confidence_threshold else "NEUTRAL",
                confidence=smoothed_conf,
                timestamp=current_time
            )

            confirmed = False
            new_token_added = None

            if agg_out.is_committed and agg_out.committed_number:
                confirmed = True
                label_text = f"Số {agg_out.committed_number}"
                self.sentence_tokens.append(label_text)
                new_token_added = label_text
                self.last_confirmed_sign = agg_out.committed_number
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

assert target_digits_block in content, "target_digits_block not found"
content = content.replace(target_digits_block, replacement_digits_block, 1)

with open('backend/vision/recognizer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Đã cập nhật backend/vision/recognizer.py thành công!")

