import json

data = json.load(open('d:/SIGN_LANGUAGE/scratch/counterfactual_simulation_results.json', encoding='utf-8'))
grid = data['sensitivity_grid']

print(f"{'mid_thresh':<12} | {'ang_thresh':<12} | {'H elim':<16} | {'V retained':<18} | {'K elim'}")
print("-" * 75)
for g in grid:
    h_str = f"{g['h_elim']}/140 ({g['h_elim_pct']:.1f}%)"
    v_str = f"{g['v_ret']}/362 ({g['v_ret_pct']:.1f}%)"
    k_str = f"{g['k_elim']}/1"
    print(f"{g['mid_thresh']:<12.2f} | {g['ang_thresh']:<12.1f} | {h_str:<16} | {v_str:<18} | {k_str}")

