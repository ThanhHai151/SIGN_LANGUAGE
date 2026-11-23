#!/usr/bin/env python3
"""Phase 0 Diagnostic Log Analyzer for QIPEDC Live-Word Recognition.

Reads JSONL logs from backend/data/diagnostics/phase0/ and produces
a detailed breakdown of recognition attempts, failure points, and statistics.

Distinguishes clearly between:
- 1 User Sign Attempt (1 Trial)
- N API calls within a trial
- M vote evaluations within a trial

Usage:
    python scratch/analyze_phase0.py [path/to/log.jsonl]
    python scratch/analyze_phase0.py --latest
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def find_latest_log():
    log_dir = Path(__file__).resolve().parents[1] / "backend" / "data" / "diagnostics" / "phase0"
    if not log_dir.is_dir():
        return None
    logs = sorted(log_dir.glob("phase0_*.jsonl"), key=os.path.getmtime)
    return logs[-1] if logs else None


def load_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no} invalid JSON: {e}", file=sys.stderr)
    return events


def analyze_log(path):
    events = load_events(path)
    if not events:
        print(f"Khong co su kien nao trong file: {path}")
        return

    print(f"\n{'='*78}")
    print(f"  BÁO CÁO PHÂN TÍCH PHASE 0 DIAGNOSTIC — WEBCAM PILOT")
    print(f"  File log: {path}")
    print(f"  Tổng số sự kiện ghi nhận: {len(events)}")
    print(f"{'='*78}\n")

    # 1. Event Distribution
    event_counts = Counter(e.get("event") for e in events)
    print("## 1. PHÂN BỐ CÁC LOẠI SỰ KIỆN")
    for ev, count in event_counts.most_common():
        print(f"  - {ev:28s}: {count:5d}")
    print()

    # 2. Trial-Based Analysis (User Sign Attempts)
    trial_starts = {e.get("trial_id"): e for e in events if e.get("event") == "trial_start" and e.get("trial_id")}
    trial_ends = [e for e in events if e.get("event") == "trial_end"]

    # Group all intermediate events by trial_id
    events_by_trial = defaultdict(list)
    for e in events:
        tid = e.get("trial_id")
        if tid:
            events_by_trial[tid].append(e)

    if trial_ends:
        print(f"{'='*78}")
        print("## 2. BẢNG CHI TIẾT TỪNG LẦN THỰC HIỆN KÝ HIỆU (USER SIGN ATTEMPTS / TRIALS)")
        print(f"{'='*78}")
        print(f"{'Trial':<6} | {'Từ dự định (Expected)':<22} | {'Lần':<5} | {'Kết quả':<12} | {'Từ nhận':<18} | {'TG(ms)':<6} | {'API':<4} | {'Vote':<4} | {'Tầng thất bại'}")
        print("-" * 110)

        first_attempts = []
        repeat_attempts = []
        latencies = []
        api_calls_list = []
        votes_list = []

        failure_layers = Counter()

        for te in trial_ends:
            tid = te.get("trial_id", "?")
            exp_word = te.get("expected_word", "?")
            exp_id = te.get("expected_id", "?")
            pred_word = te.get("predicted_word") or "-"
            outcome = te.get("outcome", "no_recognition")
            dur = te.get("duration_ms", 0)
            api_c = te.get("api_calls_count", 0)
            votes_c = te.get("votes_count", 0)
            time_confirm = te.get("time_to_confirm_ms")

            # Determine attempt number
            t_start = trial_starts.get(tid, {})
            attempt_num = t_start.get("attempt", te.get("attempt", 1))

            api_calls_list.append(api_c)
            votes_list.append(votes_c)
            if time_confirm:
                latencies.append(time_confirm)

            # Diagnose failure layer for this trial
            trial_events = events_by_trial.get(tid, [])
            motion_in_trial = [ev for ev in trial_events if ev.get("event") == "motion_gate"]
            stage1_in_trial = [ev for ev in trial_events if ev.get("event") == "stage1_coarse"]
            dtw_in_trial = [ev for ev in trial_events if ev.get("event") == "dtw_scores"]
            votes_in_trial = [ev for ev in trial_events if ev.get("event") == "vote"]
            decisions_in_trial = [ev for ev in trial_events if ev.get("event") == "match_decision"]

            failure_layer = "N/A (Đúng)"
            if outcome != "correct":
                if not any(m.get("motion_detected") for m in motion_in_trial):
                    failure_layer = "Motion Gate (Biên độ < 0.16)"
                elif stage1_in_trial and not any(s.get("expected_in_stage1") for s in stage1_in_trial):
                    failure_layer = "Stage 1 (Lọc mất từ mục tiêu)"
                elif dtw_in_trial and any(d.get("expected_scored") for d in dtw_in_trial):
                    # Check why DTW failed
                    best_dtw = min((d.get("expected_dtw_distance", 999) for d in dtw_in_trial if d.get("expected_dtw_distance") is not None), default=999)
                    if best_dtw > 0.52:
                        failure_layer = f"Stage 2 DTW (Dist {best_dtw:.3f} > 0.52)"
                    else:
                        # Distance was acceptable, was it vote dropped?
                        if votes_in_trial and any(v.get("entry_id") == exp_id and not v.get("accepted") for v in votes_in_trial):
                            failure_layer = "Frontend Vote (Chỉ đạt 1 vote rồi reset)"
                        else:
                            failure_layer = f"Stage 2 DTW (Không đủ separation margin)"
                elif votes_in_trial and any(v.get("entry_id") == exp_id and not v.get("accepted") for v in votes_in_trial):
                    failure_layer = "Frontend Vote (Chỉ đạt 1 vote rồi reset)"
                else:
                    failure_layer = "DTW/Tracking (Không khớp)"

                failure_layers[failure_layer.split(" (")[0]] += 1

            if attempt_num == 1:
                first_attempts.append(outcome == "correct")
            else:
                repeat_attempts.append(outcome == "correct")

            dur_str = f"{int(time_confirm if time_confirm else dur)}"
            print(f"{tid:<6} | {exp_word:<22} | Lần {attempt_num:<2} | {outcome:<12} | {pred_word:<18} | {dur_str:<6} | {api_c:<4} | {votes_c:<4} | {failure_layer}")

        print("-" * 110)
        print()

        # Summary of User Sign Attempts
        n_trials = len(trial_ends)
        n_correct = sum(1 for te in trial_ends if te.get("outcome") == "correct")
        n_incorrect = sum(1 for te in trial_ends if te.get("outcome") == "incorrect")
        n_no_rec = sum(1 for te in trial_ends if te.get("outcome") == "no_recognition")

        print("## 3. THỐNG KÊ TỔNG HỢP THEO LẦN THỰC HIỆN CỦA NGƯỜI DÙNG")
        print(f"  - Tổng số lần người dùng thực hiện ký hiệu (Trials): {n_trials}")
        print(f"  - Số lần xác nhận ĐÚNG (Correct):                   {n_correct} ({n_correct/n_trials*100:.1f}%)")
        print(f"  - Số lần xác nhận SAI (False Positive):              {n_incorrect} ({n_incorrect/n_trials*100:.1f}%)")
        print(f"  - Số lần KHÔNG NHẬN DIỆN (No Recognition):           {n_no_rec} ({n_no_rec/n_trials*100:.1f}%)")
        print()

        if first_attempts:
            first_corr = sum(first_attempts)
            print(f"  - TỶ LỆ ĐÚNG NGAY LẦN 1 (First-Attempt Accuracy):   {first_corr}/{len(first_attempts)} ({first_corr/len(first_attempts)*100:.1f}%)")
        if repeat_attempts:
            rep_corr = sum(repeat_attempts)
            print(f"  - TỶ LỆ ĐÚNG Ở LẦN 2-3 (Repeat Attempts):           {rep_corr}/{len(repeat_attempts)} ({rep_corr/len(repeat_attempts)*100:.1f}%)")

        if latencies:
            print(f"  - Thời gian từ khi ký đến khi xác nhận (Latency):  min={min(latencies):.0f}ms, mean={sum(latencies)/len(latencies):.0f}ms, max={max(latencies):.0f}ms")
        if api_calls_list:
            print(f"  - Số lần gọi API trung bình cho 1 lần ký:          {sum(api_calls_list)/len(api_calls_list):.1f} calls/trial")
        if votes_list:
            print(f"  - Số lượt vote trung bình cho 1 lần ký:            {sum(votes_list)/len(votes_list):.1f} votes/trial")
        print()

        if failure_layers:
            print("## 4. BÓC TÁCH NGUYÊN NHÂN THẤT BẠI THEO TỪNG TẦNG XỬ LÝ (Root Causes)")
            for fl, cnt in failure_layers.most_common():
                pct = cnt / (n_trials - n_correct) * 100 if (n_trials - n_correct) > 0 else 0
                print(f"  - {fl:30s}: {cnt:3d} lần ({pct:5.1f}% tổng số lần thất bại)")
            print()

    # 5. Continuous Pipeline Metrics (Window-level)
    print("## 5. THỐNG KÊ CHI TIẾT CÁC TẦNG KỸ THUẬT NỘI BỘ (Window-Level)")

    # Motion gate
    motion_events = [e for e in events if e.get("event") == "motion_gate"]
    if motion_events:
        passed = sum(1 for e in motion_events if e.get("motion_detected"))
        failed = len(motion_events) - passed
        print(f"  [Motion Gate] Tổng: {len(motion_events)} cửa sổ | Vượt qua: {passed} ({passed/len(motion_events)*100:.1f}%) | Chặn: {failed}")

    # Stage 1 Coarse
    stage1_events = [e for e in events if e.get("event") == "stage1_coarse"]
    if stage1_events:
        with_exp = [s for s in stage1_events if "expected_in_stage1" in s]
        if with_exp:
            in_s1 = sum(1 for s in with_exp if s.get("expected_in_stage1"))
            print(f"  [Stage 1 Coarse] Từ mục tiêu lọt vào Top-24: {in_s1}/{len(with_exp)} ({in_s1/len(with_exp)*100:.1f}%)")

    # Stage 2 DTW
    dtw_events = [e for e in events if e.get("event") == "dtw_scores"]
    if dtw_events:
        top1_dists = [e.get("top1_distance") for e in dtw_events if e.get("top1_distance") is not None]
        margins = [e.get("margin_1_2") for e in dtw_events if e.get("margin_1_2") is not None]
        if top1_dists:
            print(f"  [Stage 2 DTW] Top-1 Distance: mean={sum(top1_dists)/len(top1_dists):.4f} (<=0.35: {sum(1 for d in top1_dists if d <= 0.35)}, <=0.52: {sum(1 for d in top1_dists if d <= 0.52)})")
        if margins:
            print(f"  [Stage 2 DTW] Margin 1-2: mean={sum(margins)/len(margins):.4f} (>=0.05: {sum(1 for m in margins if m >= 0.05)})")

    # Frontend vote accumulation
    votes = [e for e in events if e.get("event") == "vote"]
    if votes:
        acc = sum(1 for v in votes if v.get("accepted"))
        rej = len(votes) - acc
        print(f"  [Frontend Vote] Đạt yêu cầu xác nhận: {acc} | Cần thêm vote (Bị chặn ở lần 1): {rej}")

    print(f"\n{'='*78}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--latest":
        log_file = Path(sys.argv[1])
    else:
        log_file = find_latest_log()

    if not log_file or not log_file.is_file():
        print("[ERROR] Không tìm thấy file log Phase 0. Hãy chạy hệ thống với $env:VSL_PHASE0_DIAGNOSTIC=\"1\" trước.")
        sys.exit(1)

    analyze_log(log_file)
