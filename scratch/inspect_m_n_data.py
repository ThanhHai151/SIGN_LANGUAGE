import json

print("=== M.json ===")
with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)
print(f"Keys: {list(m_data.keys())}")
print(f"Video ID: {m_data.get('video_id')}, Letter: {m_data.get('letter')}, Samples: {len(m_data.get('samples', []))}")
if m_data.get('samples'):
    print("Sample 0 keys:", list(m_data['samples'][0].keys()))
    phases = {}
    for s in m_data['samples']:
        phases[s.get('phase')] = phases.get(s.get('phase'), 0) + 1
    print("Phases in M.json:", phases)

print("\n=== N.json ===")
with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)
print(f"Video ID: {n_data.get('video_id')}, Letter: {n_data.get('letter')}, Samples: {len(n_data.get('samples', []))}")
phases_n = {}
for s in n_data['samples']:
    phases_n[s.get('phase')] = phases_n.get(s.get('phase'), 0) + 1
print("Phases in N.json:", phases_n)

print("\n=== real_alphabet_landmarks.json ===")
with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    fixture = json.load(f)
m_fixture = [s for s in fixture.get('samples', []) if s.get('letter_upper') == 'M']
n_fixture = [s for s in fixture.get('samples', []) if s.get('letter_upper') == 'N']
print(f"M samples in test fixture: {len(m_fixture)}")
print(f"N samples in test fixture: {len(n_fixture)}")

