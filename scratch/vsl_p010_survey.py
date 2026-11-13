"""VSL-P010: Survey toàn bộ sessions và events - Read-only."""
import json, os, glob, collections, sys

BASE = r'E:\NAM4\SEMINAR\SIGN_LANGUAGE\debug\words'
sessions = sorted(glob.glob(os.path.join(BASE, '*', 'events.jsonl')))

total_preds = 0
total_resets = 0
sessions_info = []

for s in sessions:
    session_dir = os.path.dirname(s)
    sname = os.path.basename(session_dir)
    sjson = os.path.join(session_dir, 'session.json')
    ckpt_type = 'unknown'; sha = 'unknown'; epoch = '?'; target = ''; trial = ''
    if os.path.exists(sjson):
        with open(sjson, encoding='utf-8') as f:
            d = json.load(f)
        ckpt_type = d.get('checkpoint', {}).get('checkpoint_type', 'unknown')
        sha = d.get('checkpoint', {}).get('checkpoint_sha256', 'unknown')[:8]
        epoch = d.get('checkpoint', {}).get('checkpoint_epoch', '?')
        target = d.get('target_word', '')
        trial = d.get('trial', '')
        summary = d.get('summary', {})
        verdict = summary.get('verdict', '')
        correct_w = summary.get('correct_windows', 0)
        total_w = summary.get('total_windows', 0)
        accuracy = summary.get('accuracy', 0)
        avg_fps = summary.get('avg_client_fps', '')
        avg_window_ms = summary.get('avg_window_duration_ms', '')
    preds = 0; resets = 0
    landmarks_sources = collections.Counter()
    with open(s, encoding='utf-8') as f:
        for line in f:
            try:
                ev = json.loads(line.strip())
                et = ev.get('event_type', '')
                if et == 'PREDICTION':
                    preds += 1
                    landmarks_sources[ev.get('landmarks_source', 'unknown')] += 1
                elif et == 'SEQUENCE_RESET':
                    resets += 1
            except:
                pass
    total_preds += preds
    total_resets += resets
    npy_count = len(glob.glob(os.path.join(session_dir, 'sequences', '*.npy')))
    sessions_info.append({
        'session': sname,
        'ckpt_type': ckpt_type,
        'sha': sha,
        'epoch': epoch,
        'target': target,
        'trial': trial,
        'preds': preds,
        'npy': npy_count,
        'resets': resets,
        'landmarks': dict(landmarks_sources),
        'verdict': verdict,
        'correct_windows': correct_w,
        'total_windows': total_w,
        'accuracy': accuracy,
        'avg_fps': avg_fps,
        'avg_window_ms': avg_window_ms,
    })

print(f"TOTAL sessions: {len(sessions_info)}")
print(f"TOTAL PREDICTION events: {total_preds}")
print(f"TOTAL SEQUENCE_RESET events: {total_resets}")
print()

# Phân nhóm theo checkpoint type
ckpt_groups = collections.defaultdict(list)
for s in sessions_info:
    ckpt_groups[s['ckpt_type']].append(s)

for ctype, group in ckpt_groups.items():
    total_preds_grp = sum(x['preds'] for x in group)
    total_npy_grp = sum(x['npy'] for x in group)
    print(f"=== CHECKPOINT: {ctype} ({len(group)} sessions, {total_preds_grp} preds, {total_npy_grp} npy) ===")
    for s in group:
        lm_str = str(s['landmarks'])
        print(f"  [{s['sha']}|ep{s['epoch']}] '{s['target']}' trial={s['trial']} | "
              f"preds={s['preds']} npy={s['npy']} resets={s['resets']} "
              f"verdict={s['verdict']} acc={s['accuracy']} | lm={lm_str}")
    print()

print("=== LANDMARK SOURCE BREAKDOWN ===")
all_lm = collections.Counter()
historical_lm = collections.Counter()
reconstructed_lm = collections.Counter()
for s in sessions_info:
    for lm, cnt in s['landmarks'].items():
        all_lm[lm] += cnt
        if s['ckpt_type'] == 'historical':
            historical_lm[lm] += cnt
        elif 'reconstructed' in s['ckpt_type']:
            reconstructed_lm[lm] += cnt

print(f"ALL: {dict(all_lm)}")
print(f"Historical: {dict(historical_lm)}")
print(f"Reconstructed: {dict(reconstructed_lm)}")

