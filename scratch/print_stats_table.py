import json

with open('scratch/tus_statistics.json', 'r', encoding='utf-8') as f:
    stats = json.load(f)

t_s = stats['T_hold']
u_s = stats['U_hold']
s_s = stats['S_hold']

print(f"{'Feature':<14} | {'T (mean +/- std) [min, max]':<32} | {'U (mean +/- std) [min, max]':<32} | {'S (mean +/- std) [min, max]':<32}")
print("-" * 115)

for feat in t_s.keys():
    t_v = t_s[feat]
    u_v = u_s[feat]
    s_v = s_s[feat]
    t_str = f"{t_v['mean']:.2f} +/- {t_v['std']:.2f} [{t_v['min']:.2f}, {t_v['max']:.2f}]"
    u_str = f"{u_v['mean']:.2f} +/- {u_v['std']:.2f} [{u_v['min']:.2f}, {u_v['max']:.2f}]"
    s_str = f"{s_v['mean']:.2f} +/- {s_v['std']:.2f} [{s_v['min']:.2f}, {s_v['max']:.2f}]"
    print(f"{feat:<14} | {t_str:<32} | {u_str:<32} | {s_str:<32}")

