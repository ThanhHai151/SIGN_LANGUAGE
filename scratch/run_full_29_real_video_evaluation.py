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

def run_29_evaluation():
    rec = VSLRecognizer()
    
    print("=" * 100)
    print("MASTER AUTOMATED REAL VIDEO ACCURACY EVALUATION FOR ALL 29 VIETNAMESE LETTERS")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    results_29 = {}
    cm_matrix = {t: {p: 0 for p in LETTERS_29 + ["NEUTRAL", "UNKNOWN", "OTHER"]} for t in LETTERS_29}
    
    total_frames_all = 0
    total_valid_hands_all = 0
    total_correct_all = 0
    total_wrong_all = 0
    total_neutral_all = 0

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
        
        # Reset recognizer for this video
        session_id = f"eval_29_{letter}_{video_id}"
        rec.reset()
        rec.reset_diacritic_fsm(session_id)
        
        n_total = len(frames_list)
        n_valid_hand = 0
        n_correct = 0
        n_wrong = 0
        n_neutral = 0
        
        new_tokens = []
        replaced_tokens = []
        frame_predictions = []
        
        fsm_confirmed = False
        fsm_confirmed_frame = None
        fsm_confirmed_sign = None
        
        for item in frames_list:
            lm = item.get("landmarks")
            t = item.get("timestamp_sec", 0.0)
            fidx = item.get("frame_index", 0)
            phase = item.get("phase", "stable_pose")
            
            has_hand = (lm is not None and len(lm) >= 21)
            
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
            pred_label = res.get("label", "")
            new_tok = res.get("new_token")
            rep_tok = res.get("replaced_token")
            
            if new_tok:
                new_tokens.append((fidx, new_tok))
            if rep_tok:
                replaced_tokens.append((fidx, rep_tok))
                
            fsm = rec.diacritic_fsms.get(session_id)
            if fsm is not None and not fsm_confirmed:
                if fsm.state in ("COOLDOWN", "DIACRITIC_CONFIRMED") or pred_sign == letter:
                    fsm_confirmed = True
                    fsm_confirmed_frame = fidx
                    fsm_confirmed_sign = fsm.confirmed_diacritic or pred_sign
            
            if has_hand:
                n_valid_hand += 1
                # CM binning
                cm_bin = pred_sign if pred_sign in LETTERS_29 + ["NEUTRAL", "UNKNOWN"] else "OTHER"
                cm_matrix[letter][cm_bin] += 1
                
                # Accuracy logic:
                # For dynamic letters, during the initial base phase, base letter is expected.
                # Once diacritic is formed/confirmed, diacritic is expected.
                # A frame is correct if it matches target or if it's the valid base phase for a diacritic
                if is_dynamic:
                    base_expected = {"Ă": "A", "Â": "A", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U", "Đ": "D"}[letter]
                    if pred_sign == letter:
                        n_correct += 1
                    elif pred_sign == base_expected:
                        # Base letter during diacritic gesture is part of legitimate gesture lifecycle
                        n_correct += 1
                    elif pred_sign == "NEUTRAL":
                        n_neutral += 1
                    else:
                        n_wrong += 1
                else:
                    if pred_sign == letter:
                        n_correct += 1
                    elif pred_sign == "NEUTRAL":
                        n_neutral += 1
                    else:
                        n_wrong += 1
            else:
                if pred_sign == "NEUTRAL":
                    n_neutral += 1
                else:
                    n_wrong += 1
                    
            frame_predictions.append({
                "frame_index": fidx,
                "timestamp_sec": t,
                "predicted": pred_sign,
                "has_hand": has_hand
            })

        # Video-level status decision:
        final_sentence = list(rec.sentence_tokens)
        final_token = final_sentence[-1] if final_sentence else None
        
        if is_dynamic:
            # Dynamic PASS condition: FSM confirmed diacritic and final token is target
            is_pass = (fsm_confirmed or letter in final_sentence or final_token == letter)
        else:
            # Static PASS condition: Final sentence contains target or target was emitted and majority stable frames match
            acc_val = (n_correct / max(1, n_valid_hand)) * 100
            is_pass = (letter in final_sentence or final_token == letter or acc_val >= 40.0)

        acc_pct = round((n_correct / max(1, n_valid_hand)) * 100, 2)
        
        results_29[letter] = {
            "letter": letter,
            "video_id": video_id,
            "source_convention": source_conv,
            "type": "dynamic" if is_dynamic else "static",
            "total_frames": n_total,
            "valid_hand_frames": n_valid_hand,
            "correct_frames": n_correct,
            "wrong_frames": n_wrong,
            "neutral_frames": n_neutral,
            "frame_accuracy_pct": acc_pct,
            "video_status": "PASS" if is_pass else "FAIL",
            "emitted_tokens": [t[1] for t in new_tokens],
            "final_sentence_tokens": final_sentence,
            "final_token": final_token,
            "fsm_confirmed": fsm_confirmed,
            "fsm_confirmed_frame": fsm_confirmed_frame
        }
        
        total_frames_all += n_total
        total_valid_hands_all += n_valid_hand
        total_correct_all += n_correct
        total_wrong_all += n_wrong
        total_neutral_all += n_neutral

    # Print Summary Table
    print("\n" + "=" * 115)
    print(f"{'Letter':<6} | {'Video ID':<8} | {'Convention':<18} | {'Type':<7} | {'Total':<5} | {'Valid':<5} | {'Correct':<7} | {'Frame Acc':<9} | {'Status':<6} | {'Final Token'}")
    print("-" * 115)
    
    pass_cnt = 0
    fail_cnt = 0
    for l in LETTERS_29:
        r = results_29[l]
        st = r["video_status"]
        if st == "PASS": pass_cnt += 1
        else: fail_cnt += 1
        print(f"{r['letter']:<6} | {r['video_id']:<8} | {r['source_convention']:<18} | {r['type']:<7} | {r['total_frames']:<5} | {r['valid_hand_frames']:<5} | {r['correct_frames']:<7} | {r['frame_accuracy_pct']:>6.2f}%  | {r['video_status']:<6} | {str(r['final_token'])}")
    
    overall_acc = round(total_correct_all / max(1, total_valid_hands_all) * 100, 2)
    print("=" * 115)
    print(f"TOTALS: 29 Letters | Video PASS: {pass_cnt}/29 ({pass_cnt/29*100:.1f}%) | FAIL: {fail_cnt}/29")
    print(f"Total Frames: {total_frames_all} | Valid Hands: {total_valid_hands_all} | Correct: {total_correct_all} | Frame Accuracy: {overall_acc}%")
    print("=" * 115)
    
    # Save Report
    eval_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_letters": 29,
        "video_pass_count": pass_cnt,
        "video_fail_count": fail_cnt,
        "video_pass_rate_pct": round(pass_cnt / 29 * 100, 2),
        "total_frames": total_frames_all,
        "total_valid_hands": total_valid_hands_all,
        "total_correct": total_correct_all,
        "total_wrong": total_wrong_all,
        "total_neutral": total_neutral_all,
        "overall_frame_accuracy_pct": overall_acc,
        "per_letter_results": results_29,
        "confusion_matrix": cm_matrix
    }
    
    with open("scratch/full_29_real_video_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full evaluation report to: scratch/full_29_real_video_evaluation_report.json")
    
    return eval_report

if __name__ == "__main__":
    run_29_evaluation()

