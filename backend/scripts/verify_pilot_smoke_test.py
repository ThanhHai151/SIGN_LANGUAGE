"""
VSL Connect AI — Controlled Webcam Pilot Smoke Test Verification Tool
======================================================================
Dự án: VSL Connect AI (Task 7F.6B)
Mục tiêu:
- Nghiệm thu tính toàn vẹn và chất lượng dữ liệu của file JSONL sinh ra từ Controlled Webcam Pilot.
- Đánh giá hai tầng độc lập: Frame-level và Trial-level (tránh giả định frame là mẫu độc lập thống kê).
- Kiểm tra cú pháp JSONL, session ID, định dạng trial ID, tính đơn điệu của frame_idx và timestamp, FPS.
- Xác thực tính độc lập của ground truth (expected_sign không can thiệp vào bộ phân loại).
- Hỗ trợ cả giao diện dòng lệnh (CLI) và import hàm phục vụ unit tests.
"""

import os
import re
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple


MANDATORY_TRIAL_FIELDS = [
    "timestamp",
    "session_id",
    "trial_id",
    "frame_idx",
    "expected_sign",
    "final_sign",
    "final_confidence",
    "prediction_source",
    "confirmed",
    "geometric",
    "static_head",
    "gating_status"
]


def parse_trial_id(trial_id: str, session_id: str) -> Optional[Tuple[str, int]]:
    """
    Phân tích trial_id theo định dạng thực tế từ HUD:
    {session_id}_{letter}_rep{rep_idx:02d}
    Trả về (letter, rep_idx) nếu hợp lệ, None nếu không đúng định dạng.
    """
    pattern = r"^" + re.escape(session_id) + r"_([A-Z])_rep(\d{2})$"
    m = re.match(pattern, trial_id)
    if not m:
        return None
    letter = m.group(1)
    rep_idx = int(m.group(2))
    return letter, rep_idx


def verify_pilot_file(
    filepath: str,
    expected_session_id: str,
    expected_letters: Optional[List[str]] = None,
    expected_reps: int = 2,
    min_fps: float = 10.0,
    min_frames_per_trial: int = 50,
    stats_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Xác minh tệp JSONL của phiên Controlled Webcam Pilot theo các tiêu chí nghiêm ngặt.
    """
    expected_letters = expected_letters or ["A", "B", "C"]
    expected_trial_ids = [
        f"{expected_session_id}_{letter}_rep{rep:02d}"
        for letter in expected_letters
        for rep in range(1, expected_reps + 1)
    ]

    report: Dict[str, Any] = {
        "filepath": filepath,
        "expected_session_id": expected_session_id,
        "passed": False,
        "errors": [],
        "warnings": [],
        "summary": {
            "total_records": 0,
            "syntax_valid_records": 0,
            "trial_records": 0,
            "inter_trial_records": 0,
            "completed_trials_count": 0,
            "expected_trials_count": len(expected_trial_ids),
        },
        "frame_level": {},
        "trial_level": {},
        "stats_verification": "NOT_PROVIDED"
    }

    if not os.path.exists(filepath):
        report["errors"].append(f"Tệp không tồn tại: {filepath}")
        return report

    records = []
    line_errors = []

    # 1. Kiểm tra 100% cú pháp JSONL
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            report["summary"]["total_records"] += 1
            try:
                rec = json.loads(line_str)
                records.append((line_num, rec))
                report["summary"]["syntax_valid_records"] += 1
            except json.JSONDecodeError as jde:
                line_errors.append(f"Dòng {line_num}: Lỗi cú pháp JSON ({jde})")

    if line_errors:
        report["errors"].extend(line_errors)

    if not records:
        report["errors"].append("Tệp không có bản ghi hợp lệ nào!")
        return report

    # 2. Phân loại bản ghi: Trial Records vs Inter-Trial / Idle Records
    trial_records: List[Tuple[int, Dict[str, Any]]] = []
    inter_trial_records: List[Tuple[int, Dict[str, Any]]] = []

    for line_num, rec in records:
        t_id = rec.get("trial_id")
        if t_id is not None:
            trial_records.append((line_num, rec))
        else:
            inter_trial_records.append((line_num, rec))

    report["summary"]["trial_records"] = len(trial_records)
    report["summary"]["inter_trial_records"] = len(inter_trial_records)

    # 3. Kiểm tra trường bắt buộc và Session ID cho Trial Records
    trials_data: Dict[str, List[Dict[str, Any]]] = {}

    for line_num, rec in trial_records:
        # Kiểm tra các trường bắt buộc
        missing = [field for field in MANDATORY_TRIAL_FIELDS if field not in rec]
        if missing:
            report["errors"].append(f"Dòng {line_num}: Thiếu các trường bắt buộc: {missing}")

        # Kiểm tra Session ID
        rec_sess = rec.get("session_id")
        if rec_sess != expected_session_id:
            report["errors"].append(
                f"Dòng {line_num}: Sai session_id ('{rec_sess}', kỳ vọng '{expected_session_id}')"
            )

        t_id = str(rec["trial_id"])
        trials_data.setdefault(t_id, []).append(rec)

    # 4. Kiểm tra danh mục Trial ID
    actual_trial_ids = sorted(trials_data.keys())
    report["summary"]["completed_trials_count"] = len(actual_trial_ids)

    missing_trials = set(expected_trial_ids) - set(actual_trial_ids)
    if missing_trials:
        report["errors"].append(f"Thiếu các trial kỳ vọng: {sorted(missing_trials)}")

    unexpected_trials = set(actual_trial_ids) - set(expected_trial_ids)
    if unexpected_trials:
        report["warnings"].append(f"Phát hiện các trial ngoài danh sách kỳ vọng: {sorted(unexpected_trials)}")

    # 5. Đánh giá từng Trial (Frame-level & Trial-level)
    trial_reports = {}
    all_trials_passed = True
    last_trial_end_time = -1.0

    for t_id in expected_trial_ids:
        if t_id not in trials_data:
            trial_reports[t_id] = {
                "passed": False,
                "error": "TRIAL_MISSING"
            }
            all_trials_passed = False
            continue

        t_frames = trials_data[t_id]
        parsed = parse_trial_id(t_id, expected_session_id)
        if not parsed:
            report["errors"].append(f"Định dạng trial_id không hợp lệ: {t_id}")
            all_trials_passed = False
            expected_letter = "UNKNOWN"
            rep_num = 0
        else:
            expected_letter, rep_num = parsed

        frame_indices = [f.get("frame_idx") for f in t_frames]
        timestamps = [f.get("timestamp") for f in t_frames]
        final_signs = [f.get("final_sign") for f in t_frames]
        final_confs = [f.get("final_confidence", 0.0) for f in t_frames]
        sources = [f.get("prediction_source", "none") for f in t_frames]
        confirmed_list = [bool(f.get("confirmed", False)) for f in t_frames]
        expected_signs = [f.get("expected_sign") for f in t_frames]

        trial_errors = []

        # 5a. Kiểm tra expected_sign khớp với chữ cái mục tiêu của trial
        mismatched_expected = [
            (idx, exp) for idx, exp in enumerate(expected_signs, start=1)
            if exp != expected_letter
        ]
        if mismatched_expected:
            trial_errors.append(
                f"expected_sign không khớp với chữ cái của trial '{expected_letter}' ({len(mismatched_expected)} frames)"
            )

        # 5b. Kiểm tra frame_idx (bắt đầu từ 1, tăng đơn điệu nghiêm ngặt không trùng lặp)
        if None in frame_indices:
            trial_errors.append("frame_idx chứa giá trị None")
        else:
            if len(frame_indices) != len(set(frame_indices)):
                trial_errors.append(f"Trùng lặp frame_idx trong trial: tổng {len(frame_indices)}, độc nhất {len(set(frame_indices))}")
            # Kiểm tra tính tăng đơn điệu
            non_monotonic_idx = any(
                frame_indices[i] >= frame_indices[i + 1]
                for i in range(len(frame_indices) - 1)
            )
            if non_monotonic_idx:
                trial_errors.append("frame_idx không tăng đơn điệu")
            # Semantics từ HUD: frame đầu tiên của trial bắt đầu từ 1
            if frame_indices and frame_indices[0] != 1:
                trial_errors.append(f"frame_idx không bắt đầu từ 1 (bắt đầu từ {frame_indices[0]})")

        # 5c. Kiểm tra timestamp (hợp lệ, dương, tăng đơn điệu nghiêm ngặt)
        if None in timestamps or any(not isinstance(t, (int, float)) or t <= 0 for t in timestamps):
            trial_errors.append("timestamp chứa giá trị không hợp lệ hoặc <= 0")
            fps = 0.0
            duration = 0.0
        else:
            non_monotonic_ts = any(
                timestamps[i] >= timestamps[i + 1]
                for i in range(len(timestamps) - 1)
            )
            if non_monotonic_ts:
                trial_errors.append("timestamp không tăng đơn điệu nghiêm ngặt")

            t_start = timestamps[0]
            t_end = timestamps[-1]
            duration = t_end - t_start

            # Kiểm tra thứ tự tuần tự giữa các trial
            if last_trial_end_time > 0 and t_start < last_trial_end_time:
                trial_errors.append(f"Trial bắt đầu (t={t_start}) trước khi trial trước kết thúc (t={last_trial_end_time})")
            last_trial_end_time = t_end

            fps = (len(t_frames) - 1) / duration if duration > 0 else 0.0
            if fps < min_fps:
                trial_errors.append(f"FPS thực tế ({fps:.1f}) thấp hơn ngưỡng tối thiểu ({min_fps:.1f})")

        # 5d. Kiểm tra số lượng frame
        if len(t_frames) < min_frames_per_trial:
            trial_errors.append(f"Số frame ({len(t_frames)}) ít hơn ngưỡng tối thiểu ({min_frames_per_trial})")

        # 5e. Thống kê nhận diện (Frame-level)
        match_count = sum(1 for s in final_signs if s == expected_letter)
        match_rate = (match_count / len(t_frames)) * 100.0 if t_frames else 0.0
        confirmed_count = sum(1 for c in confirmed_list if c)
        confirmed_rate = (confirmed_count / len(t_frames)) * 100.0 if t_frames else 0.0
        avg_conf = sum(final_confs) / len(final_confs) if final_confs else 0.0

        source_counts: Dict[str, int] = {}
        for src in sources:
            source_counts[src] = source_counts.get(src, 0) + 1

        sign_counts: Dict[str, int] = {}
        for s in final_signs:
            sign_counts[s] = sign_counts.get(s, 0) + 1

        # 5f. Đánh giá Trial-level: Xác định nhãn đại diện (Representative Label)
        # Bỏ qua NEUTRAL và UNKNOWN để tìm cử chỉ tích cực đa số
        non_neutral_signs = [s for s in final_signs if s not in ("NEUTRAL", "UNKNOWN", None)]
        if not non_neutral_signs:
            representative_label = "NEUTRAL"
        else:
            # Nhãn có số frame nhiều nhất trong các nhãn non-neutral
            sign_freq: Dict[str, int] = {}
            for s in non_neutral_signs:
                sign_freq[s] = sign_freq.get(s, 0) + 1
            representative_label = max(sign_freq.items(), key=lambda item: item[1])[0]

        # Tiêu chí TRIAL PASS:
        # 1. Không có lỗi kỹ thuật (frame_idx, timestamp, FPS, min_frames, session_id).
        # 2. Representative label khớp với expected_letter.
        # 3. Tỷ lệ frame đúng >= 50%.
        # 4. Có ít nhất 1 frame đạt trạng thái confirmed.
        trial_pass = (
            len(trial_errors) == 0 and
            representative_label == expected_letter and
            match_rate >= 50.0 and
            confirmed_count >= 1
        )

        if not trial_pass:
            all_trials_passed = False
            if representative_label != expected_letter:
                trial_errors.append(
                    f"Nhãn đại diện '{representative_label}' không khớp kỳ vọng '{expected_letter}'"
                )
            if match_rate < 50.0:
                trial_errors.append(f"Tỷ lệ frame khớp ({match_rate:.1f}%) < 50.0%")
            if confirmed_count < 1:
                trial_errors.append("Không có frame nào đạt trạng thái confirmed")

        trial_reports[t_id] = {
            "letter": expected_letter,
            "rep": rep_num,
            "passed": trial_pass,
            "errors": trial_errors,
            "frame_count": len(t_frames),
            "fps": round(fps, 2),
            "duration_sec": round(duration, 2),
            "frame_accuracy_pct": round(match_rate, 2),
            "confirmed_rate_pct": round(confirmed_rate, 2),
            "average_confidence": round(avg_conf, 4),
            "representative_label": representative_label,
            "sign_distribution": sign_counts,
            "source_distribution": source_counts
        }

    report["trial_level"] = trial_reports

    # 6. Thống kê toàn cục Frame-level
    all_trial_frames = [f for t_list in trials_data.values() for f in t_list]
    if all_trial_frames:
        total_matches = sum(
            1 for f in all_trial_frames
            if f.get("final_sign") == f.get("expected_sign") and f.get("expected_sign") is not None
        )
        total_confirmed = sum(1 for f in all_trial_frames if f.get("confirmed"))
        report["frame_level"] = {
            "total_trial_frames": len(all_trial_frames),
            "overall_match_rate_pct": round((total_matches / len(all_trial_frames)) * 100.0, 2),
            "overall_confirmed_rate_pct": round((total_confirmed / len(all_trial_frames)) * 100.0, 2),
        }

    # 7. Đối chiếu thống kê Telemetry (nếu được cung cấp)
    if stats_dict is not None:
        logged = stats_dict.get("logged_count")
        queued = stats_dict.get("queued_count")
        dropped_q = stats_dict.get("dropped_queue_full", 0)
        dropped_io = stats_dict.get("dropped_io_error", 0)

        stats_ok = True
        stats_notes = []

        if dropped_q > 0:
            report["errors"].append(f"Telemetry ghi nhận dropped_queue_full = {dropped_q}")
            stats_ok = False
        if dropped_io > 0:
            report["errors"].append(f"Telemetry ghi nhận dropped_io_error = {dropped_io}")
            stats_ok = False
        if logged is not None and logged != report["summary"]["total_records"]:
            report["warnings"].append(
                f"Số record trong file ({report['summary']['total_records']}) khác logged_count ({logged})"
            )
        if queued is not None and logged is not None and queued != logged:
            report["warnings"].append(f"Queued count ({queued}) != logged count ({logged})")

        report["stats_verification"] = "PASSED" if stats_ok else "FAILED"
    else:
        report["stats_verification"] = "NOT_PROVIDED (Verified purely from JSONL)"

    # 8. Kết luận nghiệm thu tổng thể
    report["passed"] = (len(report["errors"]) == 0 and all_trials_passed)
    return report


def print_verification_report(report: Dict[str, Any]):
    """In báo cáo định dạng chuẩn ra terminal."""
    print("=" * 80)
    print("      VSL CONNECT AI — CONTROLLED WEBCAM PILOT SMOKE TEST AUDIT REPORT")
    print("=" * 80)
    print(f"File:         {report['filepath']}")
    print(f"Session ID:   {report['expected_session_id']}")
    print(f"Tổng records: {report['summary']['total_records']} "
          f"(Trial frames: {report['summary']['trial_records']}, "
          f"Inter-trial/idle: {report['summary']['inter_trial_records']})")
    print(f"Trial counts: {report['summary']['completed_trials_count']}/{report['summary']['expected_trials_count']}")
    print(f"Telemetry:    {report['stats_verification']}")
    print("-" * 80)

    print("\n--- 1. TRIAL-LEVEL EVALUATION (MỨC ĐỘ THỬ NGHIỆM) ---")
    header = f"{'Trial ID':<36} | {'Chữ':<3} | {'Frames':<6} | {'FPS':<5} | {'Match%':<6} | {'Conf%':<6} | {'Nhãn ĐD':<7} | {'Kết quả':<6}"
    print(header)
    print("-" * len(header))

    for t_id, data in report["trial_level"].items():
        if not data.get("letter"):
            print(f"{t_id:<36} | {'N/A':<3} | {'0':<6} | {'0.0':<5} | {'0.0':<6} | {'0.0':<6} | {'N/A':<7} | ❌ FAIL")
            continue

        res_str = "✅ PASS" if data["passed"] else "❌ FAIL"
        row = (
            f"{t_id:<36} | "
            f"{data['letter']:<3} | "
            f"{data['frame_count']:<6} | "
            f"{data['fps']:<5.1f} | "
            f"{data['frame_accuracy_pct']:<6.1f} | "
            f"{data['confirmed_rate_pct']:<6.1f} | "
            f"{data['representative_label']:<7} | "
            f"{res_str}"
        )
        print(row)
        if data.get("errors"):
            for err in data["errors"]:
                print(f"   └─ ⚠️  {err}")

    print("\n--- 2. FRAME-LEVEL EVALUATION (MỨC ĐỘ KHUNG HÌNH) ---")
    fl = report.get("frame_level", {})
    if fl:
        print(f"- Tổng số frame trong các trial:   {fl.get('total_trial_frames')}")
        print(f"- Tỷ lệ frame dự đoán đúng:        {fl.get('overall_match_rate_pct')}%")
        print(f"- Tỷ lệ frame đạt confirmed:       {fl.get('overall_confirmed_rate_pct')}%")
    else:
        print("Không có dữ liệu frame-level hợp lệ.")

    if report["warnings"]:
        print("\n--- CẢNH BÁO (WARNINGS) ---")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    if report["errors"]:
        print("\n--- DANH SÁCH LỖI (ERRORS) ---")
        for e in report["errors"]:
            print(f"  [X] {e}")

    print("\n" + "=" * 80)
    verdict = "🎉 OVERALL VERDICT: PASSED (READY FOR CONTROLLED PILOT)" if report["passed"] else "❌ OVERALL VERDICT: FAILED"
    print(verdict)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Xác minh dữ liệu JSONL cho Controlled Webcam Pilot Smoke Test."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Đường dẫn file JSONL cần kiểm tra (hoặc dùng --file)"
    )
    parser.add_argument(
        "--file",
        dest="file_opt",
        default=None,
        help="Đường dẫn file JSONL cần kiểm tra"
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        default=None,
        help="Mã Session ID kỳ vọng (hoặc dùng --session-id)"
    )
    parser.add_argument(
        "--session-id",
        dest="session_id_opt",
        default=None,
        help="Mã Session ID kỳ vọng"
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=10.0,
        help="Ngưỡng FPS tối thiểu cho mỗi trial (mặc định: 10.0)"
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=50,
        help="Số frame tối thiểu cho mỗi trial 5s (mặc định: 50)"
    )
    parser.add_argument(
        "--letters",
        type=str,
        default="A,B,C",
        help="Danh sách chữ cái kỳ vọng, phân tách bởi dấu phẩy (mặc định: 'A,B,C')"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Số lần lặp (reps) cho mỗi chữ cái (mặc định: 2)"
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        default=None,
        help="Chuỗi JSON hoặc đường dẫn file chứa telemetry stats để đối chiếu"
    )

    args = parser.parse_args()

    file_path = args.file_opt or args.file
    session_id = args.session_id_opt or args.session_id

    if not session_id:
        # Nếu chưa truyền session_id, thử trích xuất từ tên file
        if file_path:
            base = os.path.basename(file_path)
            m = re.match(r"^pilot_([a-zA-Z0-9_-]+?)(?:_\d{8}_\d{6}_\d+)?\.jsonl$", base)
            if m:
                session_id = m.group(1)

    if not session_id:
        print("❌ Lỗi: Cần cung cấp Session ID kỳ vọng thông qua argument (ví dụ: smoke_signer01_20261101)")
        sys.exit(1)

    if not file_path:
        # Tự động tìm kiếm file trong thư mục mặc định
        default_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "diagnostics", "pilot"
        )
        candidate = os.path.join(default_dir, f"pilot_{session_id}.jsonl")
        if os.path.exists(candidate):
            file_path = candidate
        else:
            # Tìm file có timestamp nếu trùng
            found = []
            if os.path.exists(default_dir):
                for f in os.listdir(default_dir):
                    if f.startswith(f"pilot_{session_id}") and f.endswith(".jsonl"):
                        found.append(os.path.join(default_dir, f))
            if found:
                file_path = sorted(found)[-1]  # Lấy file mới nhất
            else:
                print(f"❌ Lỗi: Không tìm thấy file JSONL cho session '{session_id}' trong {default_dir}")
                sys.exit(1)

    expected_letters = [l.strip().upper() for l in args.letters.split(",") if l.strip()]

    stats_dict = None
    if args.stats_json:
        if os.path.exists(args.stats_json):
            with open(args.stats_json, "r", encoding="utf-8") as sf:
                stats_dict = json.load(sf)
        else:
            try:
                stats_dict = json.loads(args.stats_json)
            except Exception:
                pass

    report = verify_pilot_file(
        filepath=file_path,
        expected_session_id=session_id,
        expected_letters=expected_letters,
        expected_reps=args.reps,
        min_fps=args.min_fps,
        min_frames_per_trial=args.min_frames,
        stats_dict=stats_dict
    )

    print_verification_report(report)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

