import json

with open('scratch/reconciled_metrics.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

summary = data['letter_summary']

def print_ranking(title, key, reverse=True, fmt="{:.2f}%"):
    s = sorted(summary, key=lambda x: (x[key], x['frame_acc'] if key != 'frame_acc' else x['trial_acc_all']), reverse=reverse)
    print(f"\n==================== RANKING BY {title} ====================")
    for i, x in enumerate(s, 1):
        val = x[key]
        val_str = fmt.format(val)
        print(f"{i:2d}. Letter {x['letter']:2s} : {val_str} (TrAcc={x['trial_acc_all']:.1f}%, FrAcc={x['frame_acc']:.1f}%, Neu={x['neutral_rate']:.1f}%, FP={x['fp_rate']:.1f}%, ConfTr={x['confirmed_rate_all']:.1f}%)")

print_ranking("TRIAL ACCURACY (ALL 224)", "trial_acc_all")
print_ranking("TRIAL ACCURACY (RESOLVED 220)", "trial_acc_res")
print_ranking("FRAME ACCURACY", "frame_acc")
print_ranking("NEUTRAL RATE (HIGHEST REJECTION)", "neutral_rate", reverse=True)
print_ranking("FALSE POSITIVE RATE (HIGHEST MISRECOGNITION)", "fp_rate", reverse=True)
print_ranking("CONFIRMED TRIAL RATE", "confirmed_rate_all")

