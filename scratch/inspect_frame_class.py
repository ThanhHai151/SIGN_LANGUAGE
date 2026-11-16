import sys, json
sys.path.insert(0, '.')
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()
with open('backend/data/alphabet_v2/motion_sequences/Ô.json', encoding='utf-8') as f:
    data = json.load(f)

for idx in [35, 36, 37]:
    item = data['sequence'][idx]
    lms = item.get('landmarks')
    res = rec.alphabet_classifier.classify_detailed(lms)
    feat = res.get('features', {})
    print(f"Frame {idx}: i_ext={feat.get('index_extended')} m_ext={feat.get('middle_extended')} r_ext={feat.get('ring_extended')} p_ext={feat.get('pinky_extended')} d_PW={feat.get('d_PW')} d_TW={feat.get('d_TW')} d_TI={feat.get('d_TI')} d_TM={feat.get('d_TM')}")
