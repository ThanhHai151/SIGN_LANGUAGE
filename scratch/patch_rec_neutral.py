with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    c = f.read()

target = """        if not p_prim or len(p_prim) < 21:
            if is_digits_mode and hasattr(self, "digit_aggregator"):"""

replacement = """        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
            if is_digits_mode and hasattr(self, "digit_aggregator"):"""

assert target in c, "target not found"
c = c.replace(target, replacement, 1)

with open('backend/vision/recognizer.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Updated recognizer.py with prediction_history neutral on missing landmarks")

