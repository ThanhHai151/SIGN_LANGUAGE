"""
VSL Connect AI - Offline Diagnostic Replay Tool
Phát lại dữ liệu landmark thô đã thu thập qua các tầng nhận diện để chẩn đoán nguyên nhân gốc rễ mà không cần mở webcam.

Chế độ hỗ trợ:
1. Frame Mode: Phân tích từng frame độc lập (Feature Extractor, Geometry, MLP, Ensemble).
2. Session Mode: Phát lại tuần tự cả phiên làm việc (Smoothing, Cooldown, Debounce).
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from unittest.mock import patch

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.vision.recognizer import VSLRecognizer
from backend.vision.feature_extractor import _shift_and_scale, compute_finger_states


def load_diagnostic_records(filepath: str) -> List[Dict[str, Any]]:
    """Đọc và giải mã toàn bộ bản ghi JSONL từ đường dẫn tệp an toàn."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy tệp log chẩn đoán: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"[Cảnh báo] Lỗi định dạng JSON tại dòng {line_num}: {e}")
    return records


def replay_frame_by_frame(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phát lại từng frame độc lập (Stateless).
    Xác định chính xác tầng nào (Feature, Geometry, MLP, Ensemble) gây sai lệch so với expected_sign.
    """
    # Tắt ghi log trong lúc phát lại để tránh ghi đè đệ quy
    os.environ["VSL_DIAGNOSTIC"] = "0"
    rec = VSLRecognizer()
    rec.diagnostic_enabled = False

    total_frames = len(records)
    evaluated_frames = 0
    match_count = 0
    mismatch_details = []
    inconsistency_count = 0

    for idx, r in enumerate(records):
        raw_landmarks = r.get("raw_landmarks", [])
        if not raw_landmarks or len(raw_landmarks) < 21:
            continue

        evaluated_frames += 1
        expected = r.get("expected_sign")
        logged_geom = r.get("geometry", {}).get("sign")
        logged_geom_conf = r.get("geometry", {}).get("confidence", 0.0)
        logged_mlp = r.get("mlp", {}).get("sign")
        logged_mlp_conf = r.get("mlp", {}).get("confidence", 0.0)
        logged_mlp_has_weights = r.get("mlp", {}).get("has_weights", False)
        logged_final = r.get("ensemble", {}).get("sign")

        # Tái hiện dự đoán trực tiếp từ raw_landmarks
        pred = rec.predict_landmarks(landmarks_list=raw_landmarks)
        replay_final = pred["sign"]
        replay_conf = pred["confidence"]

        # Kiểm tra tính nhất quán với bản ghi log ban đầu
        is_consistent = (replay_final == logged_final)
        if not is_consistent:
            inconsistency_count += 1

        # Chẩn đoán phân loại tầng lỗi nếu có expected_sign
        if not expected:
            root_cause = "NO_GROUND_TRUTH"
        elif replay_final == expected:
            match_count += 1
            root_cause = "MATCH"
        else:
            # Điều kiện phân loại nguyên nhân chính xác theo logic ensemble:
            # 1. MLP Override: Geometry đúng nhưng bị MLP với confidence > 0.6 ghi đè
            if logged_geom == expected and logged_final != expected and logged_mlp_has_weights and logged_mlp_conf > 0.6:
                root_cause = f"MLP_OVERRIDE (Geometry đúng '{logged_geom}' [conf={logged_geom_conf}], bị MLP '{logged_mlp}' [conf={logged_mlp_conf}] ghi đè)"
            # 2. Geometry sai, MLP đúng nhưng confidence <= 0.6 nên không cứu được
            elif logged_geom != expected and logged_mlp == expected and logged_mlp_conf <= 0.6:
                root_cause = f"GEOMETRY_MISMATCH_MLP_UNDERCONFIDENT (Geometry sai '{logged_geom}', MLP đúng '{logged_mlp}' nhưng conf={logged_mlp_conf} <= 0.6)"
            # 3. Cả hai tầng Geometry và MLP đều sai
            elif logged_geom != expected and logged_mlp != expected:
                root_cause = f"BOTH_LAYERS_MISMATCH (Geometry='{logged_geom}', MLP='{logged_mlp}', Kỳ vọng='{expected}')"
            # 4. Geometry đúng nhưng Final sai do phân xử ensemble khác
            elif logged_geom == expected and logged_final != expected:
                root_cause = f"ENSEMBLE_ARBITRATION_MISMATCH (Geometry='{logged_geom}', Final='{logged_final}')"
            else:
                root_cause = "AMBIGUOUS_UNRESOLVED"

            mismatch_details.append({
                "frame_idx": r.get("frame_idx", idx),
                "expected": expected,
                "replay_final": replay_final,
                "logged_final": logged_final,
                "is_consistent": is_consistent,
                "geom_sign": logged_geom,
                "geom_conf": logged_geom_conf,
                "mlp_sign": logged_mlp,
                "mlp_conf": logged_mlp_conf,
                "root_cause": root_cause,
                "finger_states": r.get("finger_states")
            })

    accuracy = (match_count / evaluated_frames * 100) if evaluated_frames > 0 else 0.0

    return {
        "mode": "frame",
        "total_frames": total_frames,
        "evaluated_frames": evaluated_frames,
        "match_count": match_count,
        "accuracy_percent": round(accuracy, 2),
        "inconsistency_count": inconsistency_count,
        "mismatches": mismatch_details
    }


def replay_session_sequential(records: List[Dict[str, Any]], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Phát lại tuần tự theo phiên (Stateful) với giả lập thời gian thực.
    Tái hiện chính xác luồng tích lũy smoothing window, thời gian cooldown và sự kiện confirmed.
    """
    os.environ["VSL_DIAGNOSTIC"] = "0"
    
    # Lọc theo session_id nếu được chỉ định
    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]

    # Sắp xếp đúng thứ tự thời gian/frame_idx
    records.sort(key=lambda x: (x.get("session_id", ""), x.get("frame_idx", 0), x.get("timestamp", 0)))

    # Nhóm theo session
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        sid = r.get("session_id") or "default"
        sessions.setdefault(sid, []).append(r)

    results_by_session = {}

    for sid, sess_records in sessions.items():
        rec = VSLRecognizer()
        rec.diagnostic_enabled = False
        
        confirmed_tokens = []
        frame_progression = []

        for r in sess_records:
            raw_lm = r.get("raw_landmarks", [])
            if not raw_lm or len(raw_lm) < 21:
                continue

            # Sử dụng timestamp thực tế từ log để giả lập đồng hồ máy tính
            # Giúp điều kiện cooldown debounce hoạt động chính xác 100% như phiên thực tế
            frame_time = r.get("timestamp")
            if frame_time is not None:
                with patch("time.time", return_value=float(frame_time)):
                    res = rec.predict_landmarks(landmarks_list=raw_lm)
            else:
                res = rec.predict_landmarks(landmarks_list=raw_lm)

            if res["confirmed"] and res.get("new_token"):
                confirmed_tokens.append({
                    "frame_idx": r.get("frame_idx"),
                    "token": res["new_token"],
                    "sign": res["sign"],
                    "confidence": res["confidence"]
                })

            frame_progression.append({
                "frame_idx": r.get("frame_idx"),
                "sign": res["sign"],
                "confirmed": res["confirmed"],
                "expected": r.get("expected_sign")
            })

        results_by_session[sid] = {
            "total_frames": len(sess_records),
            "confirmed_tokens": confirmed_tokens,
            "final_sentence": " ".join(rec.sentence_tokens),
            "progression_sample": frame_progression[:10]
        }

    return {
        "mode": "session",
        "total_sessions": len(sessions),
        "sessions": results_by_session
    }


def print_diagnostic_report(summary: Dict[str, Any]):
    """Hiển thị báo cáo kết quả phát lại chẩn đoán ra terminal."""
    mode = summary.get("mode")
    print("=" * 75)
    print(f" BÁO CÁO PHÁT LẠI CHẨN ĐOÁN VSL (CHẾ ĐỘ: {mode.upper()})")
    print("=" * 75)

    if mode == "frame":
        print(f"Tổng số frame trong file:  {summary['total_frames']}")
        print(f"Frame hợp lệ đánh giá:    {summary['evaluated_frames']}")
        print(f"Khớp nhãn kỳ vọng:         {summary['match_count']} ({summary['accuracy_percent']}%)")
        print(f"Số frame không khớp:       {len(summary['mismatches'])}")
        print(f"Frame không nhất quán log: {summary['inconsistency_count']}")

        if summary["mismatches"]:
            print("\nChi tiết các frame dự đoán lệch tầng:")
            print("-" * 75)
            for m in summary["mismatches"][:15]:
                print(f"Frame #{m['frame_idx']}: Kỳ vọng='{m['expected']}' | Replay='{m['replay_final']}' (Log='{m['logged_final']}')")
                print(f"  -> Chi tiết: Geom='{m['geom_sign']}' [conf={m['geom_conf']}], MLP='{m['mlp_sign']}' [conf={m['mlp_conf']}]")
                print(f"  -> Phân loại: {m['root_cause']}")
                print("-" * 75)
    elif mode == "session":
        print(f"Tổng số phiên (sessions): {summary['total_sessions']}")
        for sid, sdata in summary["sessions"].items():
            print(f"\n[Phiên: {sid}] ({sdata['total_frames']} frames)")
            print(f"  -> Câu hoàn chỉnh: \"{sdata['final_sentence']}\"")
            print(f"  -> Các từ được xác nhận (Confirmed):")
            for tok in sdata["confirmed_tokens"]:
                print(f"     * Frame #{tok['frame_idx']}: '{tok['token']}' (conf: {tok['confidence']})")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="VSL Diagnostic Replay Tool")
    parser.add_argument("--file", required=True, help="Đường dẫn đến tệp log JSONL")
    parser.add_argument("--mode", choices=["frame", "session"], default="frame", help="Chế độ phát lại: frame hoặc session")
    parser.add_argument("--session-id", default=None, help="Lọc theo session ID cụ thể (ở chế độ session)")

    args = parser.parse_args()
    records = load_diagnostic_records(args.file)

    if args.mode == "frame":
        summary = replay_frame_by_frame(records)
    else:
        summary = replay_session_sequential(records, session_id=args.session_id)

    print_diagnostic_report(summary)


if __name__ == "__main__":
    main()
