import json

with open('backend/data/alphabet_v2/audit/task_7e5b_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print(f"{'Target':<6} | {'Video':<8} | {'Confirmed':<10} | {'First Frame':<12} | {'Tokens'}")
print("=" * 80)
for k, v in data.get('video_replays', {}).items():
    tokens = [t[1] for t in v.get('confirmed_tokens', [])]
    print(f"{k:<6} | {v.get('video_id'):<8} | {str(v.get('diacritic_confirmed')):<10} | {str(v.get('first_diacritic_frame')):<12} | {tokens}")

