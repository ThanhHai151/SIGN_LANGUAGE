import json

stats = json.load(open('scratch/h_v_comparison_stats.json', 'r', encoding='utf-8'))
trials_h = json.load(open('scratch/h_trial_breakdown.json', 'r', encoding='utf-8'))

# Let's inspect trial table
trial_rows = []
for tid, c in trials_h.items():
    tot = c['_total']
    v = c.get('V', 0)
    k = c.get('K', 0)
    r = c.get('R', 0)
    neu = c.get('NEUTRAL', 0)
    rep_num = tid.split('_')[4]
    trial_rows.append(f"| `{tid}` | **{rep_num}** | {tot} | **{v}** ({v/tot*100:.1f}%) | {k} ({k/tot*100:.1f}%) | {r} ({r/tot*100:.1f}%) | {neu} ({neu/tot*100:.1f}%) |")

trial_table_md = "\n".join(trial_rows)

# Comparison table
comp_rows = []
keys_order = [
    ('ang_IM_dir', 'Góc mở trỏ-giữa (độ)', '°'),
    ('mid_horiz_ratio', 'Tỷ lệ phương ngang ngón giữa', ''),
    ('idx_horiz_ratio', 'Tỷ lệ phương ngang ngón trỏ', ''),
    ('d_MW', 'Khoảng cách cổ tay - đầu ngón giữa', ''),
    ('d_IW', 'Khoảng cách cổ tay - đầu ngón trỏ', ''),
    ('d_IM', 'Khoảng cách đầu ngón trỏ - giữa', ''),
    ('diff_x_IM', 'Độ chênh lệch tọa độ X trỏ-giữa', ''),
    ('mcp_width_2d', 'Độ rộng khớp bàn tay 2D (MCP)', ''),
    ('thumb_ring_dist', 'Khoảng cách ngón cái - áp út', ''),
    ('thumb_mid_dist', 'Khoảng cách ngón cái - giữa', ''),
    ('idx_tip_y', 'Tọa độ chuẩn hóa Y đầu ngón trỏ', ''),
    ('mid_tip_y', 'Tọa độ chuẩn hóa Y đầu ngón giữa', ''),
    ('ring_curl', 'Độ gập ngón áp út (ring curl)', ''),
    ('pinky_curl', 'Độ gập ngón út (pinky curl)', '')
]

for k, label, unit in keys_order:
    d = stats.get(k, {})
    hv = d.get('H_to_V')
    hn = d.get('H_to_NEUTRAL')
    vv = d.get('V_to_V')
    
    str_hv = f"{hv['mean']:.3f} ± {hv['std']:.3f} [{hv['median']:.3f}]" if hv else "N/A"
    str_hn = f"{hn['mean']:.3f} ± {hn['std']:.3f} [{hn['median']:.3f}]" if hn else "N/A"
    str_vv = f"{vv['mean']:.3f} ± {vv['std']:.3f} [{vv['median']:.3f}]" if vv else "N/A"
    
    comp_rows.append(f"| **`{k}`** ({label}) | {str_hv} | {str_hn} | {str_vv} |")

comp_table_md = "\n".join(comp_rows)

print("Generated tables successfully!")

