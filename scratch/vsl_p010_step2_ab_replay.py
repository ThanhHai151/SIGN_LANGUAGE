"""
VSL-P010 — Script 2: Fixed-Input A/B Replay
Chạy cả 2 checkpoint (historical & reconstructed_best) trên TẤT CẢ NPY hợp lệ.
Lưu kết quả per-sequence và by-group.

Constraints: Read-only runtime code. Không modify weights.
"""
import os, sys, json, glob, csv
from collections import defaultdict
import numpy as np
import torch

PROJECT_ROOT = r'E:\NAM4\SEMINAR\SIGN_LANGUAGE'
sys.path.insert(0, PROJECT_ROOT)

from backend.models.sstcn_model import SSTCN_Attention
from backend.models.vsl_102_dictionary import SIGN_DICT_102

DEBUG_DIR  = os.path.join(PROJECT_ROOT, 'debug', 'words')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'vsl', 'word_ab_fixed_input')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hai checkpoint phân tích A/B
CHECKPOINTS = {
    'historical': {
        'sha': '2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697',
        'path': os.path.join(PROJECT_ROOT, 'backend', 'models', 'weights', 'sign_sstcn_attention_model.pth'),
        'epoch': 32,
        'label': 'historical',
    },
    'reconstructed_best': {
        'sha': '370c5edcadfbc2b0b2476b6fb7dfe7ca3d53f68f3a2fcb13036e5b7ebabf9fad',
        'path': os.path.join(PROJECT_ROOT, 'backend', 'models', 'weights', 'sign_sstcn_reconstructed_model.pth'),
        'epoch': 10,
        'label': 'reconstructed_best',
    },
}

THRESHOLD = 0.65
WORD_CLASS_MIN = 39
WORD_CLASS_MAX = 101


def load_model(ckpt_info: dict, device='cpu') -> torch.nn.Module:
    model = SSTCN_Attention(num_classes=102, num_joints=75, in_channels=3, dropout=0.3)
    ckpt = torch.load(ckpt_info['path'], map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference_words(model: torch.nn.Module, arr: np.ndarray, device='cpu'):
    """Words-mode inference: mask classes 0-38 và 102+."""
    t = arr.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,30,75)
    inp = torch.tensor(t, dtype=torch.float32).to(device)
    with torch.inference_mode():
        logits = model(inp)
        ml = logits.clone()
        ml[:, :WORD_CLASS_MIN] = -float('inf')
        if ml.shape[1] > WORD_CLASS_MAX + 1:
            ml[:, WORD_CLASS_MAX + 1:] = -float('inf')
        probs = torch.softmax(ml, dim=1).squeeze(0).cpu().numpy()
    top_k_i = np.argsort(probs)[::-1][:5]
    top_k_p = probs[top_k_i]
    return top_k_p, top_k_i, probs


def collect_all_npy_records():
    """Thu thập tất cả NPY hợp lệ với metadata từ events.jsonl."""
    records = []
    for session_json_path in sorted(glob.glob(os.path.join(DEBUG_DIR, '*', 'session.json'))):
        session_dir = os.path.dirname(session_json_path)
        with open(session_json_path, encoding='utf-8') as f:
            meta = json.load(f)
        
        events_path = os.path.join(session_dir, 'events.jsonl')
        if not os.path.exists(events_path):
            continue
        
        pred_events = []
        with open(events_path, encoding='utf-8') as f:
            for line in f:
                try:
                    ev = json.loads(line.strip())
                    if ev.get('event_type') == 'PREDICTION':
                        pred_events.append(ev)
                except:
                    pass
        
        for ev in pred_events:
            seq_file = ev.get('sequence_file', '')
            npy_path = os.path.join(session_dir, seq_file)
            if not os.path.exists(npy_path):
                continue
            arr = np.load(npy_path)
            if arr.shape != (30, 75, 3):
                continue
            if not np.all(np.isfinite(arr)):
                continue
            
            records.append({
                'session_dir': session_dir,
                'session_id': meta.get('session_id', ''),
                'target_word': meta.get('target_word', ''),
                'target_class': meta.get('target_class'),
                'trial': meta.get('trial'),
                'source_ckpt_type': meta.get('checkpoint', {}).get('checkpoint_type', 'unknown'),
                'source_ckpt_sha8': meta.get('checkpoint', {}).get('checkpoint_sha256', '')[:8],
                'window_id': ev.get('window_id'),
                'npy_path': npy_path,
                'landmarks_source': ev.get('landmarks_source', 'unknown'),
                'is_static_shoulders': ev.get('is_static_shoulders', False),
                'left_hand_missing_ratio': ev.get('left_hand_missing_ratio', 0),
                'right_hand_missing_ratio': ev.get('right_hand_missing_ratio', 0),
                'both_hands_missing_ratio': ev.get('both_hands_missing_ratio', 0),
                'total_missing_ratio': ev.get('total_missing_landmarks_ratio', 0),
                'max_consec_missing_lh': ev.get('max_consecutive_missing_lh', 0),
                'max_consec_missing_rh': ev.get('max_consecutive_missing_rh', 0),
                'shoulder_left_std': ev.get('shoulder_left_std', 0),
                'shoulder_right_std': ev.get('shoulder_right_std', 0),
                'elbow_left_std': ev.get('elbow_left_std', 0),
                'wrist_left_std': ev.get('wrist_left_std', 0),
                'window_duration_ms': ev.get('window_duration_ms'),
                'client_fps': ev.get('client_fps'),
                'inference_latency_ms': ev.get('inference_latency_ms'),
                'arr': arr,
            })
    return records


def run_ab_replay(records, models):
    """Chạy cả hai model trên từng record. Lưu kết quả per-sequence."""
    per_window = []
    for rec in records:
        arr = rec['arr']
        target_class = rec['target_class']
        
        row = {
            'session_id': rec['session_id'],
            'window_id': rec['window_id'],
            'target_word': rec['target_word'],
            'target_class': target_class,
            'source_ckpt_type': rec['source_ckpt_type'],
            'landmarks_source': rec['landmarks_source'],
            'is_static_shoulders': rec['is_static_shoulders'],
            'left_hand_missing_ratio': rec['left_hand_missing_ratio'],
            'right_hand_missing_ratio': rec['right_hand_missing_ratio'],
            'both_hands_missing_ratio': rec['both_hands_missing_ratio'],
            'total_missing_ratio': rec['total_missing_ratio'],
            'max_consec_missing_lh': rec['max_consec_missing_lh'],
            'max_consec_missing_rh': rec['max_consec_missing_rh'],
            'shoulder_left_std': rec['shoulder_left_std'],
            'shoulder_right_std': rec['shoulder_right_std'],
            'window_duration_ms': rec['window_duration_ms'],
            'client_fps': rec['client_fps'],
        }
        
        for ckpt_name, model in models.items():
            probs, indices, full_probs = run_inference_words(model, arr)
            
            top1_class = int(indices[0])
            top1_conf = float(probs[0])
            top1_label = SIGN_DICT_102.get(top1_class, 'UNKNOWN')
            
            # Target confidence: xác suất của chính từ mục tiêu
            target_conf = float(full_probs[target_class]) if target_class is not None else 0.0
            
            # Top5 list
            top5 = [(int(i), SIGN_DICT_102.get(int(i), 'UNK'), round(float(p), 4))
                    for p, i in zip(probs, indices)]
            
            # Correct = top1 là target và >= threshold
            correct = (top1_class == target_class) and (top1_conf >= THRESHOLD)
            
            # In top5?
            top5_classes = [t[0] for t in top5]
            target_in_top5 = (target_class in top5_classes)
            
            prefix = ckpt_name  # e.g. "historical", "reconstructed_best"
            row[f'{prefix}_top1_class'] = top1_class
            row[f'{prefix}_top1_label'] = top1_label
            row[f'{prefix}_top1_conf'] = round(top1_conf, 4)
            row[f'{prefix}_target_conf'] = round(target_conf, 4)
            row[f'{prefix}_correct'] = correct
            row[f'{prefix}_target_in_top5'] = target_in_top5
            row[f'{prefix}_top5'] = str(top5)
        
        # Chênh lệch dự đoán giữa hai model
        h_top1 = row.get('historical_top1_class')
        r_top1 = row.get('reconstructed_best_top1_class')
        row['models_agree'] = (h_top1 == r_top1)
        row['hist_better'] = row.get('historical_correct', False) and not row.get('reconstructed_best_correct', False)
        row['recon_better'] = row.get('reconstructed_best_correct', False) and not row.get('historical_correct', False)
        row['both_correct'] = row.get('historical_correct', False) and row.get('reconstructed_best_correct', False)
        row['both_wrong'] = not row.get('historical_correct', False) and not row.get('reconstructed_best_correct', False)
        
        per_window.append(row)
    
    return per_window


def compute_group_stats(per_window):
    """Nhóm theo (a) landmarks_source, (b) target_word, (c) session_id, (d) ckpt_tested."""
    groups = defaultdict(lambda: {'hist_correct': 0, 'recon_correct': 0,
                                   'hist_top5': 0, 'recon_top5': 0,
                                   'total': 0})
    
    # Group (a): landmarks_source
    by_lm = defaultdict(list)
    # Group (b): target_word
    by_word = defaultdict(list)
    # Group (c): session_id
    by_session = defaultdict(list)
    
    for row in per_window:
        lm = row['landmarks_source']
        word = row['target_word']
        sess = row['session_id']
        
        by_lm[lm].append(row)
        by_word[word].append(row)
        by_session[sess].append(row)
    
    summary_rows = []
    
    for group_key, group_name, group_dict in [
        ('landmarks_source', 'landmarks_source', by_lm),
        ('target_word', 'target_word', by_word),
        ('session_id', 'session_id', by_session),
    ]:
        for key, rows in sorted(group_dict.items()):
            n = len(rows)
            h_c = sum(1 for r in rows if r.get('historical_correct'))
            r_c = sum(1 for r in rows if r.get('reconstructed_best_correct'))
            h_t5 = sum(1 for r in rows if r.get('historical_target_in_top5'))
            r_t5 = sum(1 for r in rows if r.get('reconstructed_best_target_in_top5'))
            agree = sum(1 for r in rows if r.get('models_agree'))
            hist_better = sum(1 for r in rows if r.get('hist_better'))
            recon_better = sum(1 for r in rows if r.get('recon_better'))
            both_c = sum(1 for r in rows if r.get('both_correct'))
            both_w = sum(1 for r in rows if r.get('both_wrong'))
            
            # Sessions với ít nhất 1 đúng
            sessions_hist_any = len(set(r['session_id'] for r in rows if r.get('historical_correct')))
            sessions_recon_any = len(set(r['session_id'] for r in rows if r.get('reconstructed_best_correct')))
            sessions_total = len(set(r['session_id'] for r in rows))
            
            summary_rows.append({
                'group_by': group_name,
                'group_key': key,
                'n_windows': n,
                'hist_top1_correct': h_c,
                'hist_top1_acc': round(h_c/n, 4) if n else 0,
                'recon_top1_correct': r_c,
                'recon_top1_acc': round(r_c/n, 4) if n else 0,
                'hist_target_top5': h_t5,
                'hist_top5_rate': round(h_t5/n, 4) if n else 0,
                'recon_target_top5': r_t5,
                'recon_top5_rate': round(r_t5/n, 4) if n else 0,
                'models_agree': agree,
                'agree_rate': round(agree/n, 4) if n else 0,
                'hist_better_windows': hist_better,
                'recon_better_windows': recon_better,
                'both_correct_windows': both_c,
                'both_wrong_windows': both_w,
                'sessions_total': sessions_total,
                'sessions_hist_any_correct': sessions_hist_any,
                'sessions_recon_any_correct': sessions_recon_any,
            })
    
    return summary_rows


def save_per_window_csv(per_window, output_path):
    if not per_window:
        return
    # Lọc bỏ cột 'arr' (numpy array) nếu có
    fieldnames = [k for k in per_window[0].keys() if k != 'arr']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(per_window)


def save_group_csv(summary_rows, output_path):
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)


if __name__ == '__main__':
    print("=== VSL-P010 FIXED-INPUT A/B REPLAY ===")
    
    # 1. Load 2 models
    print("Loading models...")
    models = {}
    for name, info in CHECKPOINTS.items():
        print(f"  Loading {name} (SHA {info['sha'][:8]}, epoch {info['epoch']})...")
        m = load_model(info)
        models[name] = m
        print(f"  -> OK, model.eval()")
    
    # 2. Thu thập tất cả NPY
    print("Collecting NPY records...")
    records = collect_all_npy_records()
    print(f"  -> {len(records)} valid NPY records found")
    
    # 3. A/B Replay
    print("Running A/B replay on all NPY...")
    per_window = run_ab_replay(records, models)
    print(f"  -> Done. {len(per_window)} rows")
    
    # 4. Lưu per_window CSV
    per_window_csv = os.path.join(OUTPUT_DIR, 'replay_per_window.csv')
    save_per_window_csv(per_window, per_window_csv)
    print(f"  Saved: {per_window_csv}")
    
    # 5. Group stats
    print("Computing group stats...")
    group_stats = compute_group_stats(per_window)
    group_csv = os.path.join(OUTPUT_DIR, 'replay_by_source_and_word.csv')
    save_group_csv(group_stats, group_csv)
    print(f"  Saved: {group_csv}")
    
    # 6. Quick summary
    n = len(per_window)
    h_total = sum(1 for r in per_window if r.get('historical_correct'))
    r_total = sum(1 for r in per_window if r.get('reconstructed_best_correct'))
    h_top5 = sum(1 for r in per_window if r.get('historical_target_in_top5'))
    r_top5 = sum(1 for r in per_window if r.get('reconstructed_best_target_in_top5'))
    agree = sum(1 for r in per_window if r.get('models_agree'))
    hist_better = sum(1 for r in per_window if r.get('hist_better'))
    recon_better = sum(1 for r in per_window if r.get('recon_better'))
    both_c = sum(1 for r in per_window if r.get('both_correct'))
    
    print(f"\n--- TỔNG KẾT A/B TRÊN {n} TENSOR ---")
    print(f"Historical   Top-1 Correct: {h_total}/{n} = {100*h_total/n:.1f}%")
    print(f"Recon Best   Top-1 Correct: {r_total}/{n} = {100*r_total/n:.1f}%")
    print(f"Historical   Target in Top5: {h_top5}/{n} = {100*h_top5/n:.1f}%")
    print(f"Recon Best   Target in Top5: {r_top5}/{n} = {100*r_top5/n:.1f}%")
    print(f"Models agree (same Top-1):  {agree}/{n} = {100*agree/n:.1f}%")
    print(f"Historical better (paired): {hist_better}")
    print(f"Recon better (paired):      {recon_better}")
    print(f"Both correct:               {both_c}")
    
    print("\n--- PER LANDMARK_SOURCE ---")
    holistic = [r for r in per_window if r['landmarks_source'] == 'holistic_raw']
    fallback  = [r for r in per_window if r['landmarks_source'] == 'hands_fallback_pose']
    for group_name, grp in [('holistic_raw', holistic), ('hands_fallback_pose', fallback)]:
        if not grp:
            continue
        ng = len(grp)
        hg = sum(1 for r in grp if r.get('historical_correct'))
        rg = sum(1 for r in grp if r.get('reconstructed_best_correct'))
        print(f"  {group_name} (n={ng}): Historical={hg}/{ng}({100*hg/ng:.1f}%) Recon={rg}/{ng}({100*rg/ng:.1f}%)")
    
    print("\n--- PER TARGET_WORD ---")
    by_word = defaultdict(list)
    for r in per_window:
        by_word[r['target_word']].append(r)
    for word, grp in sorted(by_word.items()):
        ng = len(grp)
        hg = sum(1 for r in grp if r.get('historical_correct'))
        rg = sum(1 for r in grp if r.get('reconstructed_best_correct'))
        print(f"  '{word}' (n={ng}): Historical={hg}/{ng}({100*hg/ng:.1f}%) Recon={rg}/{ng}({100*rg/ng:.1f}%)")
    
    print(f"\nDone. Files saved in: {OUTPUT_DIR}")

