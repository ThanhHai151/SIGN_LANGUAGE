with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    c = f.read()

target_p_prim_none = """        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
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
                    }"""

replacement_p_prim_none = """        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
            if is_digits_mode and hasattr(self, "digit_aggregator"):
                agg_out = self.digit_aggregator.update("NEUTRAL", 0.0, timestamp=current_time)
                if agg_out.is_committed and agg_out.committed_number:
                    committed_str = agg_out.committed_number
                    label_text = f"Số {committed_str}"
                    if len(committed_str) > 1:
                        prefix_label = f"Số {committed_str[0]}"
                        if self.sentence_tokens and self.sentence_tokens[-1] == prefix_label:
                            self.sentence_tokens[-1] = label_text
                        else:
                            self.sentence_tokens.append(label_text)
                    elif not (self.sentence_tokens and self.sentence_tokens[-1] == label_text):
                        self.sentence_tokens.append(label_text)

                    self.last_confirmed_sign = committed_str
                    self.last_confirmed_time = current_time
                    return {
                        "sign": committed_str,
                        "label": label_text,
                        "type": "digit",
                        "confidence": 1.0,
                        "confirmed": True,
                        "new_token": label_text,
                        "sentence": " ".join(self.sentence_tokens),
                        "description": f"Chữ số {committed_str}",
                        "aggregator": agg_out.to_dict()
                    }"""

assert target_p_prim_none in c, "target_p_prim_none not found"
c = c.replace(target_p_prim_none, replacement_p_prim_none, 1)

target_digits_block = """            # Smoothing cho chữ số đơn tức thời
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
                self.last_confirmed_time = current_time"""

replacement_digits_block = """            # Smoothing cho chữ số đơn tức thời
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
                    self.last_confirmed_time = current_time"""

assert target_digits_block in c, "target_digits_block not found"
c = c.replace(target_digits_block, replacement_digits_block, 1)

with open('backend/vision/recognizer.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Updated recognizer.py with dual confirmation compatibility")

