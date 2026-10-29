import json

with open("backend/tests/fixtures/real_alphabet_landmarks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Schema version: {data.get('schema_version')}")
print(f"Target domain: {data.get('target_domain')}")
print(f"Total videos: {data.get('total_videos')}")
print(f"Total samples: {data.get('total_samples')}")
print(f"Successful samples: {data.get('successful_samples')}")
print(f"Failed samples: {data.get('failed_samples')}")

samples = data.get("samples", [])
print(f"Number of sample items: {len(samples)}")

letters = set(s['letter'] for s in samples)
print(f"Unique letters count: {len(letters)}")
print(f"Unique letters: {sorted(list(letters))}")

