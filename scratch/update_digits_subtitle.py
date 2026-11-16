# -*- coding: utf-8 -*-
path = 'backend/vision/recognizer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Sửa phần số 10
old_10_block = """            # 1. ƯU TIÊN 1: Nhận diện cử chỉ động riêng cho Số 10 VSL
            det10_res = self.digit_10_detector.update(p_prim, timestamp=current_time)
            if det10_res["detected"]:
                self.prediction_history.clear()
                label_text = "Số 10"
                tts_text = DIGIT_TTS_MAP["10"]
                
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
                    "tts_text": tts_text,
                    "type": "digit",
                    "confidence": round(float(det10_res["confidence"]), 2),
                    "confirmed": True,
                    "new_token": label_text,
                    "sentence": " ".join(self.sentence_tokens),
                    "description": "Chữ số 10",
                    "detector_10": det10_res
                }"""

new_10_block = """            # 1. ƯU TIÊN 1: Nhận diện cử chỉ động riêng cho Số 10 VSL
            det10_res = self.digit_10_detector.update(p_prim, timestamp=current_time)
            if det10_res["detected"]:
                self.prediction_history.clear()
                label_token = "10"
                tts_text = DIGIT_TTS_MAP["10"]
                
                # Phụ đề trong chế độ chữ số bỏ hoàn toàn từ "Số", chỉ hiển thị số thuần túy ("10")
                if self.sentence_tokens and (self.sentence_tokens[-1] in ("1", "Số 1")):
                    self.sentence_tokens[-1] = label_token
                elif not (self.sentence_tokens and self.sentence_tokens[-1] == label_token):
                    self.sentence_tokens.append(label_token)
                    
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
                    "label": "Số 10",
                    "tts_text": tts_text,
                    "type": "digit",
                    "confidence": round(float(det10_res["confidence"]), 2),
                    "confirmed": True,
                    "new_token": label_token,
                    "sentence": " ".join(self.sentence_tokens),
                    "description": "Chữ số 10",
                    "detector_10": det10_res
                }"""

assert old_10_block in content, "old_10_block not found"
content = content.replace(old_10_block, new_10_block, 1)

# Sửa phần số đơn 0-9
old_single_block = """                        if smoothed_digit != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                            self.last_confirmed_sign = smoothed_digit
                            self.last_confirmed_time = current_time
                            confirmed = True
                            label_text = f"Số {smoothed_digit}"
                            self.sentence_tokens.append(label_text)
                            new_token_added = label_text
                            tts_text = DIGIT_TTS_MAP.get(smoothed_digit, smoothed_digit)"""

new_single_block = """                        if smoothed_digit != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                            self.last_confirmed_sign = smoothed_digit
                            self.last_confirmed_time = current_time
                            confirmed = True
                            label_token = smoothed_digit
                            self.sentence_tokens.append(label_token)
                            new_token_added = label_token
                            tts_text = DIGIT_TTS_MAP.get(smoothed_digit, smoothed_digit)"""

assert old_single_block in content, "old_single_block not found"
content = content.replace(old_single_block, new_single_block, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated backend/vision/recognizer.py to emit pure digit tokens in subtitle sentence.")

