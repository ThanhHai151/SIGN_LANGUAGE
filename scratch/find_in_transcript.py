import json

with open(r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'evaluate_rule_m_diagnostics' in line:
            data = json.loads(line)
            content = data.get('content', '')
            print(f"Line {i} has evaluate_rule_m_diagnostics, content length: {len(content)}")
            if 'def evaluate_rule_m_diagnostics' in content:
                print("Found definition in content!")
                # Let's extract the definition
                idx = content.find('def evaluate_rule_m_diagnostics')
                print(content[idx:idx+500])
                break

