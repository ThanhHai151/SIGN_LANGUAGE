"""
VSL-P010 — Script 1: Reproducibility Check
Replay từng NPY với checkpoint gốc của session và so sánh với log.

Constraints:
- Read-only: không sửa runtime code
- Dùng đúng preprocess logic từ recognizer.py: (30,75,3) -> transpose(2,0,1) -> (1,3,30,75)
- Dùng đúng masking words: logits[:, :39] = -inf
- Chạy model.eval() + torch.no_grad()
"""
import os, sys, json, glob
import numpy as np
import torch

# Thêm project root vào path
PROJECT_ROOT = r'E:\NAM4\SEMINAR\SIGN_LANGUAGE'
sys.path.insert(0, PROJECT_ROOT)

from backend.models.sstcn_model import load_sstcn_model, SSTCN_Attention
from backend.models.vsl_102_dictionary import SIGN_DICT_102

DEBUG_DIR   = os.path.join(PROJECT_ROOT, 'debug', 'words')
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'docs', 'vsl', 'word_ab_fixed_input')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_SHA_TO_PATH = {
    '2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697':
        os.path.join(PROJECT_ROOT, 'backend', 'models', 'weights', 'sign_sstcn_attention_model.pth'),
    '370c5edcadfbc2b0b2476b6fb7dfe7ca3d53f68f3a2fcb13036e5b7ebabf9fad':
        os.path.join(PROJECT_ROOT, 'backend', 'models', 'weights', 'sign_sstcn_reconstructed_model.pth'),
    '301dc6bcf0d282acf029ad9c8c921766de6da51d5167f7becafbba630ae319ec':
        os.path.join(PROJECT_ROOT, 'backend', 'models', 'weights', 'sign_sstcn_reconstructed_last.pth'),
}

CKPT_LABEL = {
    '2a8ddcc4': 'historical',
    '370c5edc': 'reconstructed_best',
    '301dc6bc': 'reconstructed_last',
}

THRESHOLD_DEFAULT = 0.65

def load_model_from_sha(full_sha: str, device='cpu') -> torch.nn.Module:
    """Load checkpoint bằng SHA-256 đầy đủ, model.eval()"""
    path = CKPT_SHA_TO_PATH.get(full_sha)
    if path is None:
        raise ValueError(f"Không tìm thấy path cho SHA {full_sha[:8]}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File không tồn tại: {path}")
    
    model = SSTCN_Attention(num_classes=102, num_joints=75, in_channels=3, dropout=0.3)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model: torch.nn.Module, npy_arr: np.ndarray, device='cpu'):
    """
    Chạy inference đúng logic runtime (recognizer.py:119-138):
    arr (30,75,3) -> transpose(2,0,1) -> (3,30,75) -> (1,3,30,75) -> model
    Words masking: logits[:, :39] = -inf
    """
    arr = npy_arr.astype(np.float32)
    tensor_seq = arr.transpose(2, 0, 1)         # (3, 30, 75)
    tensor_seq = tensor_seq[np.newaxis, ...]     # (1, 3, 30, 75)
    tensor_input = torch.tensor(tensor_seq, dtype=torch.float32).to(device)
    
    with torch.inference_mode():
        logits = model(tensor_input)
        # Words masking (recognizer.py:127-131)
        masked_logits = logits.clone()
        masked_logits[:, :39] = -float('inf')
        if masked_logits.shape[1] > 102:
            masked_logits[:, 102:] = -float('inf')
        probs = torch.softmax(masked_logits, dim=1).squeeze(0)
        top_k_probs, top_k_indices = torch.topk(probs, k=min(5, len(probs)))
    
    top_k_probs = top_k_probs.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    
    return top_k_probs, top_k_indices


def collect_sessions():
    sessions = []
    for session_json_path in sorted(glob.glob(os.path.join(DEBUG_DIR, '*', 'session.json'))):
        session_dir = os.path.dirname(session_json_path)
        with open(session_json_path, encoding='utf-8') as f:
            meta = json.load(f)
        
        # Đọc events
        events_path = os.path.join(session_dir, 'events.jsonl')
        pred_events = []
        if os.path.exists(events_path):
            with open(events_path, encoding='utf-8') as f:
                for line in f:
                    try:
                        ev = json.loads(line.strip())
                        if ev.get('event_type') == 'PREDICTION':
                            pred_events.append(ev)
                    except:
                        pass
        
        sessions.append({
            'session_dir': session_dir,
            'session_id': meta.get('session_id', ''),
            'target_word': meta.get('target_word', ''),
            'target_class': meta.get('target_class'),
            'trial': meta.get('trial'),
            'ckpt': meta.get('checkpoint', {}),
            'pred_events': pred_events,
        })
    return sessions


def replay_session(session, model, threshold=THRESHOLD_DEFAULT):
    """
    Replay từng NPY của session với model đã nạp.
    Trả về list dict kết quả.
    """
    results = []
    for ev in session['pred_events']:
        seq_file = ev.get('sequence_file', '')
        npy_path = os.path.join(session['session_dir'], seq_file)
        
        if not os.path.exists(npy_path):
            results.append({
                'npy_found': False,
                'session_id': session['session_id'],
                'window_id': ev.get('window_id'),
                'error': f'NPY not found: {npy_path}',
            })
            continue
        
        arr = np.load(npy_path)
        # Kiểm tra shape, dtype, finite
        shape_ok = arr.shape == (30, 75, 3)
        dtype_ok = arr.dtype == np.float32
        finite_ok = bool(np.all(np.isfinite(arr)))
        
        if not shape_ok:
            results.append({
                'npy_found': True,
                'session_id': session['session_id'],
                'window_id': ev.get('window_id'),
                'error': f'Bad shape: {arr.shape}',
                'shape_ok': False,
            })
            continue
        
        top_k_probs, top_k_indices = run_inference(model, arr)
        
        replay_top1_class = int(top_k_indices[0])
        replay_top1_conf = float(top_k_probs[0])
        replay_top1_label = SIGN_DICT_102.get(replay_top1_class, 'UNKNOWN')
        
        # So sánh với log gốc
        log_top1_class = ev.get('top1_class')
        log_top1_conf = ev.get('top1_confidence')
        log_top1_label = ev.get('top1_label', '')
        
        top1_match = (replay_top1_class == log_top1_class)
        conf_diff = abs(replay_top1_conf - (log_top1_conf or 0))
        
        # So sánh top5
        log_top5_classes = [t.get('class') for t in ev.get('top_5', [])]
        replay_top5_classes = [int(x) for x in top_k_indices]
        top5_match = (log_top5_classes == replay_top5_classes)
        
        # target confidence
        target_class = session['target_class']
        target_conf_replay = 0.0
        for prob, idx in zip(top_k_probs, top_k_indices):
            if int(idx) == target_class:
                target_conf_replay = float(prob)
                break
        
        log_target_conf = ev.get('target_confidence', 0)
        target_conf_diff = abs(target_conf_replay - (log_target_conf or 0))
        
        replay_correct = (replay_top1_class == target_class) and (replay_top1_conf >= threshold)
        
        results.append({
            'npy_found': True,
            'session_id': session['session_id'],
            'window_id': ev.get('window_id'),
            'target_class': target_class,
            'target_word': session['target_word'],
            'shape_ok': shape_ok,
            'dtype_ok': dtype_ok,
            'finite_ok': finite_ok,
            'arr_min': float(arr.min()),
            'arr_max': float(arr.max()),
            # Log gốc
            'log_top1_class': log_top1_class,
            'log_top1_label': log_top1_label,
            'log_top1_conf': log_top1_conf,
            'log_target_conf': log_target_conf,
            'log_prediction_correct': ev.get('prediction_correct', False),
            # Replay
            'replay_top1_class': replay_top1_class,
            'replay_top1_label': replay_top1_label,
            'replay_top1_conf': round(replay_top1_conf, 4),
            'replay_target_conf': round(target_conf_replay, 4),
            'replay_correct': replay_correct,
            # Match
            'top1_class_match': top1_match,
            'top5_order_match': top5_match,
            'top1_conf_diff': round(conf_diff, 5),
            'target_conf_diff': round(target_conf_diff, 5),
            # Landmark context
            'landmarks_source': ev.get('landmarks_source', 'unknown'),
            'is_static_shoulders': ev.get('is_static_shoulders', False),
            'left_hand_missing_ratio': ev.get('left_hand_missing_ratio', 0),
            'right_hand_missing_ratio': ev.get('right_hand_missing_ratio', 0),
            'both_hands_missing_ratio': ev.get('both_hands_missing_ratio', 0),
            'total_missing_ratio': ev.get('total_missing_landmarks_ratio', 0),
            'window_duration_ms': ev.get('window_duration_ms'),
            'client_fps': ev.get('client_fps'),
            'threshold': threshold,
            # Checkpoint logging
            'ckpt_sha_session': session['ckpt'].get('checkpoint_sha256', '')[:8],
            'ckpt_type_session': session['ckpt'].get('checkpoint_type', 'unknown'),
        })
    return results


if __name__ == '__main__':
    print("=== VSL-P010 REPRODUCIBILITY CHECK ===")
    sessions = collect_sessions()
    print(f"Sessions trouvées: {len(sessions)}")
    
    # Tải model theo SHA
    models_cache = {}
    
    all_results = []
    for session in sessions:
        full_sha = session['ckpt'].get('checkpoint_sha256', '')
        if not full_sha or full_sha not in CKPT_SHA_TO_PATH:
            print(f"  SKIP {session['session_id']}: SHA không nhận ra ({full_sha[:8]})")
            continue
        
        if full_sha not in models_cache:
            print(f"  Loading model SHA {full_sha[:8]}...")
            try:
                models_cache[full_sha] = load_model_from_sha(full_sha)
                print(f"  -> OK")
            except Exception as ex:
                print(f"  -> FAIL: {ex}")
                continue
        
        model = models_cache[full_sha]
        print(f"  Replaying {session['session_id']} ({len(session['pred_events'])} events)...")
        results = replay_session(session, model)
        all_results.extend(results)
    
    # Tổng kết reproducibility
    valid = [r for r in all_results if r.get('shape_ok', False) and 'error' not in r]
    total = len(valid)
    top1_matched = sum(1 for r in valid if r['top1_class_match'])
    top5_matched = sum(1 for r in valid if r['top5_order_match'])
    # Tolerance 1%
    within_1pct = sum(1 for r in valid if r['top1_conf_diff'] <= 0.01)
    within_5pct = sum(1 for r in valid if r['top1_conf_diff'] <= 0.05)
    npy_missing = sum(1 for r in all_results if not r.get('npy_found', True))
    
    print(f"\n--- TỔNG KẾT REPRODUCIBILITY ---")
    print(f"Total NPY replay: {total}")
    print(f"Top-1 class khớp: {top1_matched}/{total} ({100*top1_matched/total:.1f}%)")
    print(f"Top-5 order khớp: {top5_matched}/{total} ({100*top5_matched/total:.1f}%)")
    print(f"Confidence diff <= 1%: {within_1pct}/{total}")
    print(f"Confidence diff <= 5%: {within_5pct}/{total}")
    print(f"NPY không tìm thấy: {npy_missing}")
    
    # Lưu kết quả vào JSON để dùng cho bước sau
    out_path = os.path.join(OUTPUT_DIR, '_reproducibility_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nKết quả lưu tại: {out_path}")
    
    # Hiển thị các sai khác
    mismatches = [r for r in valid if not r['top1_class_match']]
    if mismatches:
        print(f"\n--- SAI KHÁC TOP-1 ({len(mismatches)} cases) ---")
        for m in mismatches[:10]:
            print(f"  {m['session_id']} w{m['window_id']}: "
                  f"log={m['log_top1_class']}({m['log_top1_label']}) "
                  f"replay={m['replay_top1_class']}({m['replay_top1_label']})")
    else:
        print("\nKhông có sai khác Top-1. Log TÁI HIỆN HOÀN TOÀN.")

