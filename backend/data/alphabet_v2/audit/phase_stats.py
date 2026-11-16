import sys
import json
import os

sys.stdout.reconfigure(encoding='utf-8')

motion_dir = 'backend/data/alphabet_v2/motion_sequences'
rep_file = 'backend/data/alphabet_v2/audit/phase_review/phase_review_report.json'

with open(rep_file, 'r', encoding='utf-8') as fp:
    rep = json.load(fp)

letters = rep['letters']
total_base_ok = 0
total_motion_ok = 0
total_entry_ok = 0
total_exit_ok = 0

per_letter_breakdown = {}

for l, d in letters.items():
    fpath = os.path.join(motion_dir, f"{l}.json")
    with open(fpath, 'r', encoding='utf-8') as fp:
        m_data = json.load(fp)
    seq = m_data['sequence']
    v = d['task5a_verified']
    
    def parse_r(r_str):
        parts = r_str.split('-')
        return int(parts[0]), int(parts[1])
    
    e_s, e_e = parse_r(v['entry']['range'])
    b_s, b_e = parse_r(v['base_pose']['range'])
    m_s, m_e = parse_r(v['diacritic_motion']['range'])
    x_s, x_e = parse_r(v['exit']['range'])
    
    cnt_e = sum(1 for s in seq if e_s <= s['frame_index'] <= e_e and s['quality_status'] == 'ok')
    cnt_b = sum(1 for s in seq if b_s <= s['frame_index'] <= b_e and s['quality_status'] == 'ok')
    cnt_m = sum(1 for s in seq if m_s <= s['frame_index'] <= m_e and s['quality_status'] == 'ok')
    cnt_x = sum(1 for s in seq if x_s <= s['frame_index'] <= x_e and s['quality_status'] == 'ok')
    
    tot_ok = sum(1 for s in seq if s['quality_status'] == 'ok')
    base_l = d['base_letter']
    
    per_letter_breakdown[l] = {
        'base_letter': base_l,
        'total_ok': tot_ok,
        'entry_ok': cnt_e,
        'base_pose_ok': cnt_b,
        'diacritic_motion_ok': cnt_m,
        'exit_ok': cnt_x
    }
    
    total_base_ok += cnt_b
    total_motion_ok += cnt_m
    total_entry_ok += cnt_e
    total_exit_ok += cnt_x
    
    print(f"{l} (Base {base_l}): Tot OK = {tot_ok:2d} | Entry = {cnt_e:2d} | BasePose = {cnt_b:2d} | Diacritic = {cnt_m:2d} | Exit = {cnt_x:2d}")

print("-" * 75)
print(f"Tổng 7 chữ: Tot OK = {sum(x['total_ok'] for x in per_letter_breakdown.values())} | Entry = {total_entry_ok} | BasePose = {total_base_ok} | Diacritic = {total_motion_ok} | Exit = {total_exit_ok}")

