"""
Controlled Pilot V3 Recognition Adapter.
Pure read-only adapter connecting to the existing VSLRecognizer and AlphabetClassifier.
Does NOT modify thresholds, classifiers, or models.
Collects raw landmarks, extracted geometric features, per-layer diagnostics,
and sets explicit NOT_EXECUTED states when layers are bypassed.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.vision.controlled_pilot_v3.constants import (
    HandState,
    RecognitionState,
    VIETNAMESE_ALPHABET_SET,
)
from backend.vision.feature_extractor import normalize_hand_landmarks


class ControlledPilotRecognitionAdapter:
    def __init__(self, recognizer: Any):
        """
        Takes the existing shared VSLRecognizer instance without mutating its architecture.
        """
        self.recognizer = recognizer

    def evaluate_frame(
        self,
        landmarks: Optional[List[List[float]]],
        expected_letter: str,
        handedness: str = "Right",
        tracking_confidence: Optional[float] = None,
        handedness_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates one frame through the existing recognition pipeline read-only.
        Returns full diagnostics for tracking, geometric features, rule diagnostics,
        MLP diagnostics, and ensemble decision.
        """
        # 1. Hand Tracking Analysis
        hand_detected = False
        valid_landmarks = False
        hand_state = HandState.NO_HAND
        clean_landmarks: Optional[List[List[float]]] = None
        landmark_quality: Optional[Dict[str, Any]] = None

        if landmarks is not None and len(landmarks) > 0:
            hand_detected = True
            try:
                arr = np.asarray(landmarks, dtype=np.float32)
                if arr.shape == (21, 3) and np.isfinite(arr).all():
                    valid_landmarks = True
                    hand_state = HandState.HAND_DETECTED
                    clean_landmarks = [[float(p[0]), float(p[1]), float(p[2])] for p in arr]
                    # Quality: bounded within [0, 1] approximately, palm span finite
                    wrist = arr[0]
                    middle_mcp = arr[9]
                    palm_span = float(np.linalg.norm(wrist - middle_mcp))
                    landmark_quality = {
                        "palm_span": round(palm_span, 4),
                        "is_finite": True,
                        "point_count": 21,
                        "tracking_confidence": tracking_confidence,
                        "handedness_score": handedness_score,
                    }
                else:
                    hand_state = HandState.LANDMARKS_INVALID
                    landmark_quality = {
                        "error": "Shape is not (21, 3) or contains non-finite values",
                        "shape": list(arr.shape),
                    }
            except Exception as e:
                hand_state = HandState.LANDMARKS_INVALID
                landmark_quality = {"error": str(e)}

        hand_tracking = {
            "hand_detected": hand_detected,
            "handedness": handedness,
            "hand_landmarks": clean_landmarks,
            "landmark_quality": landmark_quality,
            "tracking_status": hand_state.value,
        }

        # 2. If no valid landmarks, return neutral diagnostic record
        if not valid_landmarks or clean_landmarks is None:
            return {
                "hand_tracking": hand_tracking,
                "geometric_features": None,
                "geometric_prediction": None,
                "geometric_rule_status": "NO_HAND" if not hand_detected else "INVALID_LANDMARKS",
                "rule_diagnostics": None,
                "failed_conditions": None,
                "mlp_prediction": None,
                "mlp_confidence": None,
                "mlp_status": "NOT_EXECUTED",
                "mlp_not_executed_reason": "No valid hand landmarks available",
                "ensemble_prediction": "NEUTRAL",
                "final_prediction": "NEUTRAL",
                "final_confidence": 0.0,
                "prediction_source": "none",
                "gating_status": "none",
                "recognition_state": RecognitionState.NEUTRAL.value,
                "smoothing_state": self._get_smoothing_snapshot(),
                "cooldown_state": self._get_cooldown_snapshot(),
            }

        # 3. Geometric Feature Extraction & Classification
        geometric_features: Optional[Dict[str, Any]] = None
        geometric_prediction: Optional[str] = None
        geometric_rule_status: str = "UNKNOWN"
        rule_diagnostics: Optional[Dict[str, Any]] = None
        failed_conditions: Optional[Any] = None

        classifier = getattr(self.recognizer, "alphabet_classifier", None)
        alpha_res: Dict[str, Any] = {}

        if classifier is not None:
            try:
                alpha_res = classifier.classify_detailed(clean_landmarks)
                geometric_rule_status = alpha_res.get("status", "UNKNOWN")
                if geometric_rule_status == "MATCH":
                    geometric_prediction = alpha_res.get("letter")
                geometric_features = alpha_res.get("features")
                rule_diagnostics = {
                    "reason": alpha_res.get("reason"),
                    "matched_candidates": alpha_res.get("matched_candidates"),
                }
                failed_conditions = alpha_res.get("failed_conditions")
            except Exception as e:
                geometric_rule_status = "ERROR"
                rule_diagnostics = {"error": str(e)}

        # 4. Static Head (MLP) Evaluation
        mlp_prediction: Optional[str] = None
        mlp_confidence: Optional[float] = None
        mlp_status: str = "NOT_EXECUTED"
        mlp_not_executed_reason: Optional[str] = None
        head_top_k: Optional[List[Dict[str, Any]]] = None

        # Check if geometric rule already matched conclusively
        if geometric_rule_status == "MATCH" and geometric_prediction in VIETNAMESE_ALPHABET_SET:
            mlp_status = "NOT_EXECUTED"
            mlp_not_executed_reason = f"Geometric rule matched '{geometric_prediction}' conclusively; MLP layer bypassed"
            final_pred = geometric_prediction
            final_conf = float(alpha_res.get("confidence", 0.95))
            pred_source = "geometric_rule"
            gating_status = "passed_geometric"
        else:
            # Execute Static Head MLP if available
            static_head = getattr(self.recognizer, "static_alphabet_head", None)
            has_static_head = getattr(self.recognizer, "has_static_head", False) and (static_head is not None)

            if has_static_head:
                try:
                    feat_72, _ = normalize_hand_landmarks(clean_landmarks)
                    head_res = static_head.predict_feature(feat_72[:72], confidence_threshold=0.0)
                    mlp_prediction = head_res.get("raw_predicted_letter", "UNKNOWN")
                    mlp_confidence = float(head_res.get("confidence", 0.0))
                    head_top_k = head_res.get("top_k", [])
                    mlp_status = "EXECUTED"

                    # Check gating against recognizer thresholds
                    req_threshold = getattr(self.recognizer, "supplemental_thresholds", {}).get(mlp_prediction, 0.70)
                    fallback_base_thresh = getattr(self.recognizer, "fallback_base_threshold", 0.80)
                    allowed_fallback = getattr(self.recognizer, "ALLOWED_FALLBACK_BASE_LETTERS", {"A", "D", "E"})
                    allowed_supp = getattr(self.recognizer, "ALLOWED_SUPPLEMENTAL_LETTERS", {"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"})

                    if mlp_prediction in allowed_fallback and mlp_confidence >= fallback_base_thresh:
                        final_pred = mlp_prediction
                        final_conf = mlp_confidence
                        pred_source = "static_head_fallback"
                        gating_status = "passed_fallback"
                    elif mlp_prediction in allowed_supp and mlp_confidence >= req_threshold:
                        final_pred = mlp_prediction
                        final_conf = mlp_confidence
                        pred_source = "static_head"
                        gating_status = "passed_supplemental"
                    else:
                        final_pred = "NEUTRAL"
                        final_conf = 0.0
                        pred_source = "none"
                        gating_status = "rejected"
                except Exception as e:
                    mlp_status = "ERROR"
                    mlp_not_executed_reason = str(e)
                    final_pred = "NEUTRAL"
                    final_conf = 0.0
                    pred_source = "none"
                    gating_status = "error"
            else:
                mlp_status = "NOT_EXECUTED"
                mlp_not_executed_reason = "Static alphabet head is not loaded or not configured"
                final_pred = "NEUTRAL"
                final_conf = 0.0
                pred_source = "none"
                gating_status = "none"

        # 5. Determine Recognition State relative to expected_letter
        if expected_letter not in VIETNAMESE_ALPHABET_SET:
            rec_state = RecognitionState.ERROR
        else:
            # Check if this letter is supported by the engine
            allowed_letters = getattr(self.recognizer, "ALLOWED_LETTERS", VIETNAMESE_ALPHABET_SET)
            if expected_letter not in allowed_letters:
                rec_state = RecognitionState.UNSUPPORTED
            elif final_pred == expected_letter:
                rec_state = RecognitionState.MATCH
            elif final_pred == "NEUTRAL":
                rec_state = RecognitionState.NEUTRAL
            elif final_pred in ("UNKNOWN", None):
                rec_state = RecognitionState.UNKNOWN
            else:
                rec_state = RecognitionState.MISMATCH

        return {
            "hand_tracking": hand_tracking,
            "geometric_features": geometric_features,
            "geometric_prediction": geometric_prediction,
            "geometric_rule_status": geometric_rule_status,
            "rule_diagnostics": rule_diagnostics,
            "failed_conditions": failed_conditions,
            "mlp_prediction": mlp_prediction,
            "mlp_confidence": mlp_confidence,
            "mlp_status": mlp_status,
            "mlp_not_executed_reason": mlp_not_executed_reason,
            "mlp_top_k": head_top_k,
            "ensemble_prediction": final_pred,
            "final_prediction": final_pred,
            "final_confidence": round(float(final_conf), 4),
            "prediction_source": pred_source,
            "gating_status": gating_status,
            "recognition_state": rec_state.value,
            "smoothing_state": self._get_smoothing_snapshot(),
            "cooldown_state": self._get_cooldown_snapshot(),
        }

    def _get_smoothing_snapshot(self) -> Dict[str, Any]:
        history = list(getattr(self.recognizer, "prediction_history", []))
        return {
            "window_size": getattr(self.recognizer, "smoothing_window", 5),
            "history_len": len(history),
            "recent_signs": [s for s, _ in history],
        }

    def _get_cooldown_snapshot(self) -> Dict[str, Any]:
        last_sign = getattr(self.recognizer, "last_confirmed_sign", None)
        last_time = getattr(self.recognizer, "last_confirmed_time", 0.0)
        cooldown_sec = getattr(self.recognizer, "cooldown_sec", 1.0)
        return {
            "last_confirmed_sign": last_sign,
            "last_confirmed_time": last_time,
            "cooldown_sec": cooldown_sec,
        }
