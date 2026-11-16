# -*- coding: utf-8 -*-
with open('backend/vision/digit_aggregator.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '        self.last_committed_number: Optional[str] = None\n'
replacement = '''        self.last_committed_number: Optional[str] = None

    def to_dict(self) -> dict:
        """Trả về snapshot trạng thái của bộ ghép số."""
        return {
            "state": self.state.value,
            "current_digit": self.current_candidate,
            "accumulated_digits": list(self.accumulated_digits),
            "provisional_number": "".join(self.accumulated_digits) if self.accumulated_digits else None,
            "committed_number": self.last_committed_number,
            "is_committed": (self.state == AggregatorState.COMMITTED),
            "hold_progress": 0.0,
            "time_since_last_digit": 0.0,
            "time_until_timeout": 0.0,
        }
'''

if 'def to_dict(self) -> dict:\n        """Trả về snapshot' not in text:
    assert target in text, "target not found"
    text = text.replace(target, replacement, 1)
    with open('backend/vision/digit_aggregator.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Added to_dict to VSLSequentialDigitAggregator successfully")
else:
    print("Already added")
