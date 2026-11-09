import unittest
from unittest.mock import patch

import numpy as np
import torch

from backend.models.alphabet_static_head import (
    DEFAULT_HEAD_WEIGHTS_PATH,
    STATIC_CLASSES_22,
    load_static_alphabet_head,
)


class TestStaticHeadSafeLoad(unittest.TestCase):
    def test_checkpoint_load_uses_weights_only_scoped_allowlist_and_infers(self):
        user_globals_before = list(torch.serialization.get_safe_globals()) if hasattr(torch.serialization, "get_safe_globals") else None

        with patch("backend.models.alphabet_static_head.torch.load", wraps=torch.load) as load_spy:
            head, metadata = load_static_alphabet_head(DEFAULT_HEAD_WEIGHTS_PATH)

        self.assertTrue(load_spy.call_args.kwargs.get("weights_only", False) or not hasattr(torch.serialization, "safe_globals"))
        self.assertTrue(load_spy.call_args.kwargs["map_location"] == "cpu")
        self.assertFalse(head.training)
        self.assertEqual(metadata["architecture"], "VSLStaticAlphabetHead (Linear 72->64, LN, LeakyReLU, Dropout 0.15; Linear 64->32, LN, LeakyReLU, Dropout 0.1; Linear 32->22)")

        result = head.predict_feature(np.zeros(72, dtype=np.float32), confidence_threshold=0.0)
        self.assertIn(result["raw_predicted_letter"], STATIC_CLASSES_22)
        self.assertTrue(np.isfinite(result["confidence"]))
        self.assertEqual(len(result["top_k"]), 5)

        if hasattr(torch.serialization, "get_safe_globals"):
            user_globals_after = list(torch.serialization.get_safe_globals())
            self.assertEqual(user_globals_after, user_globals_before)
        self.assertNotIn(scalar_global, user_globals_after)
        self.assertNotIn(dtype_global, user_globals_after)
        self.assertNotIn(float64_global, user_globals_after)


if __name__ == "__main__":
    unittest.main()
