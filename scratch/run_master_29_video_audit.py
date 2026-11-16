import os
import sys
import json
import time
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["VSL_DIACRITIC_DEBUG"] = "1"
os.environ["VSL_PREFETCH"] = "0"

from backend.vision.recognizer import VSLRecognizer

LETTERS_29 = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
]

DYNAMIC_LETTERS = {'Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ'}

METADATA_PATH = "backend/data/alphabet_v2/metadata/videos_metadata.json"
with open(METADATA_PATH, encoding="utf-8") as f:
    video_meta = json.load(f)

def run_comprehensive_audit():
    rec = VSLRecognizer()
    
    print("=" * 110)
    print("RIGOROUS 29-LETTER REAL VIDEO ACCURACY, LATENCY & PIPELINE INTEGRITY AUDIT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

    results_29 = {}
    cm_matrix = {t: {p: 0 for p in LETTERS_29 + ["NEUTRAL", "UNKNOWN", "OTHER"]} for t in LETTERS_29}
    
    total_frames_all = 0
    total_valid_hands_all = 0
    total_tracking_loss_all = 0
    total_correct_all = 0
    total_wrong_all = 0
    total_neutral_all = 0
    
    tts_results_7 = {}

    for letter in LETTERS_29:
        is_dynamic = letter in DYNAMIC_LETTERS
        m_path = f"backend/data/alphabet_v2/motion_sequences/{letter}.json"
        s_path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
        
        file_path = m_path if is_dynamic else s_path
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            
        frames_list = data.get("sequence" if is_dynamic else "samples", [])
        video_id = data.get("video_id", video_meta.get(letter, {}).get("video_id", "unknown"))
        meta_entry = video_meta.get(letter, {})
        source_conv = meta_entry.get("source_convention", "VERIFIED_SOUTHERN" if "N" in video_id else "VERIFIED_NATIONAL")
        
        # Reset recognizer for this video trial
        session_id = f"audit_29_{letter}_{video_id}"
        rec.reset()
        rec.reset_diacritic_fsm(session_id)
        
        n_total = len(frames_list)
        n_valid_hand = 0
        n_tracking_loss = 0
        n_correct = 0
        n_wrong = 0
        n_neutral = 0
        
        # Phase tracking
        stable_total = 0
        stable_correct = 0
        stable_neutral = 0
        stable_wrong = 0
        
        new_tokens = []
        replaced_tokens = []
        spoken_tts = []
        
        first_token_frame = None
        first_token_time = None
        first_target_token_frame = None
        first_target_token_time = None
        
        fsm_confirmed = False
        fsm_confirmed_frame = None
        fsm_confirmed_sign = None
        
        for item in frames_list:
            lm = item.get("landmarks")
            t = item.get("timestamp_sec", 0.0)
            fidx = item.get("frame_index", 0)
            phase = item.get("phase", "stable_pose")
            is_stable_phase = (phase == "stable_pose" or (is_dynamic and 15 <= fidx <= n_total - 10))
            
            has_hand = (lm is not None and len(lm) >= 21 and item.get("quality_status") == "ok")
            
            if not has_hand:
                n_tracking_loss += 1
            else:
                n_valid_hand += 1
                
            res = rec.predict_landmarks(
                lm,
                mode="letters",
                session_id=session_id,
                trial_id=session_id,
                timestamp=t,
                frame_idx=fidx,
                expected_sign=letter,
                phase=phase
            )
            
            pred_sign = res.get("sign", "NEUTRAL")
            new_tok = res.get("new_token")
            rep_tok = res.get("replaced_token")
            tts_spk = res.get("tts_speak")
            
            if new_tok:
                new_tokens.append((fidx, t, new_tok))
                if first_token_frame is None:
                    first_token_frame = fidx
                    first_token_time = t
                if new_tok == letter and first_target_token_frame is None:
                    first_target_token_frame = fidx
                    first_target_token_time = t
                    
            if rep_tok:
                replaced_tokens.append((fidx, t, rep_tok, new_tok))
            if tts_spk:
                spoken_tts.append((fidx, t, tts_spk))
                
            fsm = rec.diacritic_fsms.get(session_id)
            if fsm is not None and not fsm_confirmed:
                if fsm.state in ("COOLDOWN", "DIACRITIC_CONFIRMED") or pred_sign == letter:
                    fsm_confirmed = True
                    fsm_confirmed_frame = fidx
                    fsm_confirmed_sign = fsm.confirmed_diacritic or pred_sign
            
            # Ground truth classification evaluation
            if has_hand:
                cm_bin = pred_sign if pred_sign in LETTERS_29 + ["NEUTRAL", "UNKNOWN"] else "OTHER"
                cm_matrix[letter][cm_bin] += 1
                
                is_frame_correct = False
                if is_dynamic:
                    base_expected = {"Ă": "A", "Â": "A", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U", "Đ": "D"}[letter]
                    if pred_sign == letter:
                        is_frame_correct = True
                        n_correct += 1
                    elif pred_sign == base_expected and not fsm_confirmed:
                        # Legitimate base pose before diacritic execution
                        is_frame_correct = True
                        n_correct += 1
                    elif pred_sign == "NEUTRAL":
                        n_neutral += 1
                    else:
                        n_wrong += 1
                else:
                    if pred_sign == letter:
                        is_frame_correct = True
                        n_correct += 1
                    elif pred_sign == "NEUTRAL":
                        n_neutral += 1
                    else:
                        n_wrong += 1
                        
                if is_stable_phase:
                    stable_total += 1
                    if is_frame_correct:
                        stable_correct += 1
                    elif pred_sign == "NEUTRAL":
                        stable_neutral += 1
                    else:
                        stable_wrong += 1
            else:
                if pred_sign == "NEUTRAL":
                    n_neutral += 1
                else:
                    n_wrong += 1

        # Video-level assessment
        final_sentence = list(rec.sentence_tokens)
        final_token = final_sentence[-1] if final_sentence else None
        
        # Token correctness:
        # A video is strictly correct if final sentence has exactly [letter]
        strictly_correct_sentence = (final_sentence == [letter])
        
        # Wrong extra tokens:
        wrong_extra_tokens = [tok[2] for tok in new_tokens if tok[2] != letter and not (is_dynamic and tok[2] == {"Ă": "A", "Â": "A", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U", "Đ": "D"}[letter])]
        
        # Video Success decision:
        if is_dynamic:
            video_pass = (letter in final_sentence and final_token == letter and fsm_confirmed)
        else:
            video_pass = (letter in final_sentence and final_token == letter and len(wrong_extra_tokens) == 0)

        frame_acc_pct = round(n_correct / max(1, n_valid_hand) * 100, 2)
        stable_acc_pct = round(stable_correct / max(1, stable_total) * 100, 2)
        tracking_loss_pct = round(n_tracking_loss / max(1, n_total) * 100, 2)
        
        latency_sec = round(first_target_token_time, 2) if first_target_token_time is not None else None
        latency_frames = first_target_token_frame if first_target_token_frame is not None else None

        results_29[letter] = {
            "letter": letter,
            "video_id": video_id,
            "source_convention": source_conv,
            "type": "dynamic" if is_dynamic else "static",
            "total_frames": n_total,
            "valid_hand_frames": n_valid_hand,
            "tracking_loss_frames": n_tracking_loss,
            "tracking_loss_pct": tracking_loss_pct,
            "correct_frames": n_correct,
            "wrong_frames": n_wrong,
            "neutral_frames": n_neutral,
            "frame_accuracy_pct": frame_acc_pct,
            "stable_pose_frames": stable_total,
            "stable_pose_correct": stable_correct,
            "stable_pose_neutral": stable_neutral,
            "stable_pose_wrong": stable_wrong,
            "stable_pose_accuracy_pct": stable_acc_pct,
            "latency_sec": latency_sec,
            "latency_frame": latency_frames,
            "emitted_tokens": [t[2] for t in new_tokens],
            "replaced_tokens": replaced_tokens,
            "spoken_tts": [s[2] for s in spoken_tts],
            "wrong_extra_tokens": wrong_extra_tokens,
            "final_sentence": final_sentence,
            "strictly_correct_sentence": strictly_correct_sentence,
            "video_pass": video_pass,
            "fsm_confirmed": fsm_confirmed,
            "fsm_confirmed_frame": fsm_confirmed_frame
        }
        
        if is_dynamic:
            tts_results_7[letter] = {
                "emitted": [t[2] for t in new_tokens],
                "spoken": [s[2] for s in spoken_tts],
                "replaced": [r[2] for r in replaced_tokens],
                "final_sentence": final_sentence
            }
            
        total_frames_all += n_total
        total_valid_hands_all += n_valid_hand
        total_tracking_loss_all += n_tracking_loss
        total_correct_all += n_correct
        total_wrong_all += n_wrong
        total_neutral_all += n_neutral

    # Print Full Rigorous Table
    print("\n" + "=" * 130)
    print(f"{'Letter':<6} | {'Video ID':<8} | {'Type':<7} | {'Total':<5} | {'Loss%':<6} | {'FrmAcc%':<7} | {'StbAcc%':<7} | {'Latency':<8} | {'Tokens Emitted':<20} | {'Final Sent':<12} | {'Status'}")
    print("-" * 130)
    
    pass_cnt = 0
    strict_cnt = 0
    for l in LETTERS_29:
        r = results_29[l]
        st = "PASS" if r["video_pass"] else "FAIL"
        if r["video_pass"]: pass_cnt += 1
        if r["strictly_correct_sentence"]: strict_cnt += 1
        lat_str = f"{r['latency_sec']}s (f{r['latency_frame']})" if r['latency_sec'] is not None else "N/A"
        toks_str = ", ".join(r["emitted_tokens"])[:18]
        sent_str = " ".join(r["final_sentence"])
        print(f"{r['letter']:<6} | {r['video_id']:<8} | {r['type']:<7} | {r['total_frames']:<5} | {r['tracking_loss_pct']:>5.1f}% | {r['frame_accuracy_pct']:>6.2f}% | {r['stable_pose_accuracy_pct']:>6.2f}% | {lat_str:<8} | {toks_str:<20} | {sent_str:<12} | {st}")
        
    overall_frame_acc = round(total_correct_all / max(1, total_valid_hands_all) * 100, 2)
    overall_loss_rate = round(total_tracking_loss_all / max(1, total_frames_all) * 100, 2)
    
    print("=" * 130)
    print(f"OVERALL SUMMARY:")
    print(f"  - Video-level Success Rate: {pass_cnt}/29 ({pass_cnt/29*100:.2f}%)")
    print(f"  - Strictly Correct Final Sentence (no extra tokens): {strict_cnt}/29 ({strict_cnt/29*100:.2f}%)")
    print(f"  - Total Frames: {total_frames_all} | Valid Hand: {total_valid_hands_all} | Tracking Loss: {total_tracking_loss_all} ({overall_loss_rate}%)")
    print(f"  - Overall Frame-level Accuracy: {total_correct_all}/{total_valid_hands_all} ({overall_frame_acc}%)")
    print("=" * 130)

    # 7 Diacritic TTS Audit
    print("\n--- 7 DIACRITIC LETTERS TTS & TOKEN REPLACEMENT AUDIT ---")
    for d_let, d_info in tts_results_7.items():
        print(f"  Letter {d_let}: Emitted={d_info['emitted']} | Spoken TTS={d_info['spoken']} | Final Sentence={d_info['final_sentence']}")

    # Save detailed report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_letters": 29,
        "video_pass_count": pass_cnt,
        "strictly_correct_count": strict_cnt,
        "video_pass_rate_pct": round(pass_cnt / 29 * 100, 2),
        "total_frames": total_frames_all,
        "total_valid_hands": total_valid_hands_all,
        "total_tracking_loss": total_tracking_loss_all,
        "tracking_loss_rate_pct": overall_loss_rate,
        "total_correct": total_correct_all,
        "total_wrong": total_wrong_all,
        "total_neutral": total_neutral_all,
        "overall_frame_accuracy_pct": overall_frame_acc,
        "per_letter_results": results_29,
        "tts_results_7": tts_results_7,
        "confusion_matrix": cm_matrix
    }
    
    with open("scratch/full_29_real_video_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full evaluation report to: scratch/full_29_real_video_evaluation_report.json")
    
    return full_report

if __name__ == "__main__":
    run_comprehensive_audit()

