"""Closed-loop self-evaluation and confusable-pair separation audit for QIPEDC dictionary videos.

Feeds extracted 75-joint QIPEDC video sequences directly into the live webcam recognition pipeline
(`match_live_words` and `compare_recording`) as if the signer in each dictionary video were standing
in front of the camera, verifying:
  1. Closed-loop recognition accuracy (> 95% target across extracted QIPEDC videos).
  2. Practice mode (`compare_recording`) self-match rate (> 98% target).
  3. Clear pairwise separation on visually similar / confusable VSL actions (e.g., 'chào' vs 'tuân lệnh',
     'chúng tôi' 1-5 người, 'khả năng' vs 'kỹ năng', 'năn nỉ' vs 'cầu nguyện' vs 'vái lạy' vs 'xin lỗi',
     'mấy giờ?', 'em gái', 'chỉ') with zero per-sign hardcoded rules.

Usage:
    python -m backend.training.audit_qipedc_self_eval --scope all
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from backend.models.lexicon import load_lexicon
from backend.training.build_qipedc_live_index import _existing_npy, select_entries
from backend.vision import qipedc_live, qipedc_practice

CONFUSABLE_GROUPS = [
    ("Chào vs Tuân lệnh", ["q-chao", "q-tuan-lenh", "q-chao-2"]),
    (
        "Đại từ Chúng tôi (1–5 người)",
        [
            "q-chung-toi-dai-tu",
            "q-chung-toi-2-nguoi-dai-tu",
            "q-chung-toi-3-nguoi-dai-tu",
            "q-chung-toi-4-nguoi-dai-tu",
            "q-chung-toi-5-nguoi-dai-tu",
        ],
    ),
    ("Khả năng vs Kỹ năng", ["q-kha-nang", "q-ky-nang"]),
    ("Chắp tay: Năn nỉ / Cầu nguyện / Vái lạy / Xin lỗi", ["q-nan-ni", "q-cau-nguyen", "q-vai-lay", "q-xin-loi"]),
    ("Hỏi giờ & Chỉ hướng & Gia đình", ["q-may-gio", "q-chi", "q-em-gai"]),
]


def evaluate_confusable_groups(by_id, vid_of):
    reports = []
    for title, group_ids in CONFUSABLE_GROUPS:
        avail = []
        for eid in group_ids:
            vid = vid_of.get(eid)
            npy = _existing_npy(vid) if vid else None
            if npy is not None and eid in by_id:
                avail.append((eid, by_id[eid]["word"], vid, npy))
        if len(avail) < 2:
            continue

        group_pass = 0
        rows = []
        for eid, word, vid, npy in avail:
            arr = np.load(npy, allow_pickle=False)
            trace = {}
            res = qipedc_live.match_live_words(arr, 1.0, [], vocab_scope="all", trace=trace)
            pred_id = res.get("entry_id") if res and res.get("status") == "word_matched" else None
            ok = pred_id == eid or (pred_id is not None and vid_of.get(pred_id) == vid)
            if ok:
                group_pass += 1
            cands = trace.get("top_candidates") or []
            top1_d = cands[0]["distance"] if cands else None
            # Find closest rival inside the confusable group
            rec = qipedc_live._variants(arr, 1.0)
            rival_best = None
            rival_word = None
            if rec is not None:
                for r_eid, r_word, r_vid, r_npy in avail:
                    if r_eid == eid or r_vid == vid:
                        continue
                    r_ref = qipedc_live._variants(np.load(r_npy, allow_pickle=False), 1.0, reference=True)
                    if r_ref is not None and r_ref[0].shape[2] == rec[0].shape[2]:
                        d = float(qipedc_live._variant_distances(rec, r_ref).min())
                        if rival_best is None or d < rival_best:
                            rival_best = d
                            rival_word = r_word
            rows.append({
                "entry_id": eid,
                "word": word,
                "video_id": vid,
                "predicted_id": pred_id,
                "predicted_word": by_id[pred_id]["word"] if pred_id in by_id else None,
                "matched": ok,
                "self_distance": top1_d,
                "closest_group_rival": rival_word,
                "rival_distance": round(rival_best, 4) if rival_best is not None else None,
                "margin_vs_group_rival": (
                    round(rival_best - (top1_d or 0.0), 4) if rival_best is not None else "different_hand_count"
                ),
            })
        reports.append({
            "group": title,
            "passed": group_pass,
            "total": len(avail),
            "accuracy_pct": round(100.0 * group_pass / len(avail), 2),
            "items": rows,
        })
    return reports


def _eval_single_item(task):
    eid, vid, word, npy_str, vocab_mode = task
    arr = np.load(npy_str, allow_pickle=False)
    prac = qipedc_practice.compare_recording(vid, arr, aspect_ratio=1.0)
    prac_matched = bool(prac.get("matched"))
    trace = {}
    res = qipedc_live.match_live_words(arr, 1.0, [], vocab_scope=vocab_mode, trace=trace)
    st = res["status"] if res else "unmatched"
    pred = res.get("entry_id") if res and res.get("status") == "word_matched" else None
    top2 = [(c["entry_id"], c["distance"]) for c in (trace.get("top_candidates") or [])[:2]]
    verdict = trace.get("verdict") or "unmatched"
    return eid, vid, word, prac_matched, st, pred, top2, verdict


def run_audit(scope="all", output_json: Path | None = None, workers: int = 0):
    t0 = time.perf_counter()
    lex = load_lexicon()
    by_id = lex["by_id"]
    idx = qipedc_live._active_index()
    indexed_set = idx["id_set"] if idx is not None else frozenset()

    entries = select_entries(scope)
    vid_of = {eid: vid for eid, vid, _ in select_entries("all")}

    vid_to_eids = defaultdict(set)
    for eid, vid, _ in select_entries("all"):
        vid_to_eids[vid].add(eid)

    vocab_mode = "common" if scope == "common" else "all"
    tasks = []
    for eid, vid, word in entries:
        if eid not in indexed_set:
            continue
        npy = _existing_npy(vid)
        if npy is not None:
            tasks.append((eid, vid, word, str(npy), vocab_mode))

    live_exact = 0
    live_video_equiv = 0
    practice_ok = 0
    status_breakdown = Counter()
    sample_mismatches = []

    if workers and workers > 1 and len(tasks) > 32:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_eval_single_item, tasks, chunksize=16))
    else:
        results = [_eval_single_item(t) for t in tasks]

    for eid, vid, word, prac_matched, st, pred, top2, verdict in results:
        if prac_matched:
            practice_ok += 1
        status_breakdown[st] += 1

        if st == "word_matched" and pred is not None:
            if pred == eid:
                live_exact += 1
                live_video_equiv += 1
            elif pred in vid_to_eids[vid]:
                live_video_equiv += 1
            elif len(sample_mismatches) < 20:
                sample_mismatches.append({
                    "entry_id": eid,
                    "word": word,
                    "video_id": vid,
                    "status": "wrong_word",
                    "predicted": pred,
                    "predicted_word": by_id.get(pred, {}).get("word"),
                })
        elif st == "ambiguous_word":
            if len(top2) >= 2 and top2[0][0] in vid_to_eids[vid] and top2[1][0] in vid_to_eids[vid]:
                live_video_equiv += 1
            elif len(sample_mismatches) < 20:
                sample_mismatches.append({
                    "entry_id": eid,
                    "word": word,
                    "video_id": vid,
                    "status": "ambiguous_word",
                    "top2": top2,
                })
        elif len(sample_mismatches) < 20:
            sample_mismatches.append({
                "entry_id": eid,
                "word": word,
                "video_id": vid,
                "status": verdict,
            })

    total = max(1, len(tasks))
    confusable_report = evaluate_confusable_groups(by_id, vid_of)
    elapsed = round(time.perf_counter() - t0, 2)

    summary = {
        "scope": scope,
        "indexed_total": len(indexed_set),
        "evaluated_videos": len(tasks),
        "elapsed_sec": elapsed,
        "practice_self_match_count": practice_ok,
        "practice_self_match_pct": round(100.0 * practice_ok / total, 2),
        "live_exact_entry_count": live_exact,
        "live_exact_entry_pct": round(100.0 * live_exact / total, 2),
        "live_action_match_count": live_video_equiv,
        "live_action_match_pct": round(100.0 * live_video_equiv / total, 2),
        "status_breakdown": dict(status_breakdown),
        "confusable_groups": confusable_report,
        "sample_mismatches": sample_mismatches,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["common", "all"], default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "diagnostics" / "qipedc_self_eval_report.json",
    )
    args = parser.parse_args()
    summary = run_audit(scope=args.scope, output_json=args.output, workers=args.workers)
    print(
        f"[audit] Evaluated {summary['evaluated_videos']} videos (scope={args.scope}) in {summary['elapsed_sec']}s:\n"
        f"  - Practice self-match: {summary['practice_self_match_count']}/{summary['evaluated_videos']} "
        f"({summary['practice_self_match_pct']}%)\n"
        f"  - Open-vocab action match: {summary['live_action_match_count']}/{summary['evaluated_videos']} "
        f"({summary['live_action_match_pct']}%) [exact entry: {summary['live_exact_entry_pct']}%]\n"
        f"  - Status breakdown: {summary['status_breakdown']}",
        flush=True,
    )
    for grp in summary["confusable_groups"]:
        print(f"  - Group [{grp['group']}]: {grp['passed']}/{grp['total']} ({grp['accuracy_pct']}%)", flush=True)


if __name__ == "__main__":
    main()
