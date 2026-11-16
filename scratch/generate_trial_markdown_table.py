import json

data = json.load(open('scratch/reconciled_metrics.json', 'r', encoding='utf-8'))
trials = data['all_trials']

header = """| Trial ID | Exp | Att | Total | Correct | Incor | Neutral | Non-Neut | Non-Neut Acc | Confirmed | Adjudication |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"""

rows = []
for t in trials:
    rows.append(
        f"| `{t['trial_id']}` | **{t['expected_sign']}** | {t['attempt']} | {t['total_frames']} | "
        f"{t['correct_frames']} | {t['incorrect_frames']} | {t['neutral_frames']} | "
        f"{t['non_neutral_frames']} | {t['correct_rate_among_non_neutral']:.1f}% | "
        f"{t['confirmed_count']} | **{t['adjudicated_result']}** |"
    )

table_content = header + "\n" + "\n".join(rows)

with open('scratch/trial_table.md', 'w', encoding='utf-8') as f:
    f.write(table_content)

print(f"Generated markdown table with {len(rows)} rows, size: {len(table_content)} bytes")

