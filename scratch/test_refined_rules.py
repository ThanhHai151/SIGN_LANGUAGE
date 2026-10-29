import os
import sys

sys.path.insert(0, os.path.abspath("."))
site_packages = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from backend.vision.feature_extractor import calculate_distance, calculate_angle, extract_dual_hand_features

def evaluate_refined_rules(feat):
    matches = []
    t_ext = feat["thumb_ext"]
    i_ext = feat["index_ext"]
    m_ext = feat["middle_ext"]
    r_ext = feat["ring_ext"]
    p_ext = feat["pinky_ext"]

    d_ti = feat["d_TI"]
    d_tm = feat["d_TM"]
    d_tw = feat["d_TW"]
    d_im = feat["d_IM"]
    d_iw = feat["d_IW"]
    d_mw = feat["d_MW"]
    d_pw = feat["d_PW"]
    diff_x_im = feat["diff_x_IM"]
    ang_ti = feat["ang_TI"]
    ang_tp = feat["ang_TP"]
    thumb_lat = feat["thumb_lat"]

    # --- B ---
    if i_ext and m_ext and r_ext and p_ext:
        if d_iw >= 1.65 and d_mw >= 1.70 and thumb_lat <= 0.35:
            matches.append(("B", "4 ngón duỗi thẳng khép sát, cái khép", 0.95))

    # --- I ---
    if p_ext and not i_ext and not m_ext and not r_ext and not t_ext:
        if d_pw >= 1.35 and d_iw <= 0.85 and d_mw <= 0.75:
            matches.append(("I", "Chỉ ngón út duỗi thẳng", 0.95))

    # --- Y ---
    if p_ext and not i_ext and not m_ext and not r_ext:
        if d_pw >= 1.35 and d_tw >= 1.05 and d_iw <= 0.85 and d_mw <= 0.75:
            matches.append(("Y", "Ngón cái và út cùng duỗi thẳng (Shaka)", 0.95))

    # --- V ---
    if i_ext and m_ext and not r_ext and not p_ext:
        if d_iw >= 1.70 and d_mw >= 1.70 and d_im >= 0.28 and diff_x_im >= 0.18:
            matches.append(("V", "Trỏ và Giữa xòe chữ V", 0.95))

    # --- D ---
    if i_ext and not m_ext and not r_ext and not p_ext:
        if d_iw >= 1.70 and d_ti <= 1.28 and d_tm <= 0.30:
            matches.append(("D", "Chỉ ngón trỏ duỗi thẳng, cái khép chạm giữa", 0.95))

    # --- L ---
    if i_ext and not m_ext and not r_ext and not p_ext:
        if d_iw >= 1.70 and d_ti >= 1.40 and d_tm >= 0.45 and thumb_lat >= 0.30 and ang_ti >= 35.0:
            matches.append(("L", "Trỏ và Cái mở vuông góc L", 0.95))

    # --- T ---
    # Ngón cái kẹp giữa trỏ và giữa, đầu trỏ bắt chéo qua ngón cái
    if d_ti <= 0.16 and d_tm >= 0.80 and d_im >= 0.80 and d_iw <= 1.30:
        matches.append(("T", "Ngón cái kẹp giữa trỏ và giữa (chữ T)", 0.95))

    # --- O ---
    # Các ngón cong tròn chụm chạm sát đầu ngón cái
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if d_ti <= 0.22 and d_tm <= 0.20 and d_iw >= 0.95 and d_mw >= 0.90:
            matches.append(("O", "Vòng tròn kín chữ O", 0.95))

    # --- C ---
    # Vòm khuyên cong mở rộng chữ C (không phải nắm đấm, không duỗi thẳng quá mức)
    if (0.55 <= d_ti <= 0.85 and 0.50 <= d_tm <= 0.85 and 
        1.35 <= d_iw <= 1.65 and 1.20 <= d_mw <= 1.55 and 1.05 <= d_pw <= 1.50):
        matches.append(("C", "Vòm cong mở chữ C", 0.95))

    # --- A ---
    # Nắm đấm ngón cái dựng đứng thẳng cạnh bàn tay
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if d_tw >= 1.32 and d_tm >= 0.75 and d_iw <= 0.90 and d_mw <= 0.75:
            matches.append(("A", "Nắm tay ngón cái dựng đứng", 0.95))

    # --- S ---
    # Nắm đấm tròn, ngón cái vắt ngang mặt trước 4 ngón
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if d_tw <= 0.90 and 0.25 <= d_ti <= 0.45 and 0.40 <= d_tm <= 0.65 and d_iw <= 0.90 and d_mw <= 0.75:
            matches.append(("S", "Nắm đấm tròn ngón cái vắt ngang", 0.95))

    # --- E ---
    # 4 ngón cong gập tì lên đầu ngón cái
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if d_ti <= 0.35 and d_tm <= 0.38 and d_tw <= 0.95 and d_iw >= 1.05 and d_mw >= 1.05:
            matches.append(("E", "4 ngón tì gập lên ngón cái chữ E", 0.95))

    return matches

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Helper trích xuất đặc trưng
def extract_feat(landmarks):
    _, meta = extract_dual_hand_features(landmarks, None)
    states = meta.get("primary_states", {})
    p = landmarks
    wrist = p[0]
    sh = [[p[i][0] - wrist[0], p[i][1] - wrist[1], p[i][2] - wrist[2]] for i in range(21)]
    palm_scale = calculate_distance(sh[0], sh[9])
    if palm_scale < 1e-5:
        return None
    return {
        "thumb_ext": bool(states.get("thumb_extended", False)),
        "index_ext": bool(states.get("index_extended", False)),
        "middle_ext": bool(states.get("middle_extended", False)),
        "ring_ext": bool(states.get("ring_extended", False)),
        "pinky_ext": bool(states.get("pinky_extended", False)),
        "ang_TI": float(calculate_angle(sh[4], sh[0], sh[8])),
        "ang_TP": float(calculate_angle(sh[4], sh[0], sh[20])),
        "thumb_lat": float(abs(sh[4][0] - sh[0][0]) / palm_scale),
        "d_TI": float(calculate_distance(sh[4], sh[8]) / palm_scale),
        "d_TM": float(calculate_distance(sh[4], sh[12]) / palm_scale),
        "d_TW": float(calculate_distance(sh[4], sh[0]) / palm_scale),
        "d_IM": float(calculate_distance(sh[8], sh[12]) / palm_scale),
        "d_IW": float(calculate_distance(sh[8], sh[0]) / palm_scale),
        "d_MW": float(calculate_distance(sh[12], sh[0]) / palm_scale),
        "d_PW": float(calculate_distance(sh[20], sh[0]) / palm_scale),
        "diff_x_IM": float(abs(sh[8][0] - sh[12][0]) / palm_scale)
    }

target_letters = ['A', 'B', 'C', 'D', 'E', 'I', 'L', 'O', 'S', 'T', 'V', 'Y']
supported = {l: {"total": 0, "correct": 0, "nomatch": 0, "conflict": 0, "wrong": 0} for l in target_letters}
unsupported = {"total": 0, "unknown": 0, "fp": 0, "fp_list": []}

for s in data['samples']:
    if s.get('quality_status') != 'ok':
        continue
    l = s['letter_upper']
    feat = extract_feat(s['landmarks'])
    if not feat:
        continue
    matches = evaluate_refined_rules(feat)
    if len(matches) == 1:
        pred = matches[0][0]
    elif len(matches) > 1:
        pred = "CONFLICT"
    else:
        pred = "UNKNOWN"

    phase = s.get('phase', 'unknown')

    if l in target_letters:
        supported[l]["total"] += 1
        if pred == l:
            supported[l]["correct"] += 1
        elif pred == "UNKNOWN":
            supported[l]["nomatch"] += 1
            print(f"[MISS] {l} frame {s['frame_idx']} ({phase})")
        elif pred == "CONFLICT":
            supported[l]["conflict"] += 1
            print(f"[CONFLICT] {l} frame {s['frame_idx']} ({phase})")
        else:
            supported[l]["wrong"] += 1
            print(f"[WRONG] {l} frame {s['frame_idx']} ({phase}) -> {pred}")
    else:
        unsupported["total"] += 1
        if pred == "UNKNOWN":
            unsupported["unknown"] += 1
        else:
            unsupported["fp"] += 1
            unsupported["fp_list"].append((l, pred, s['frame_idx'], phase))

print("\n" + "="*50)
for l in target_letters:
    r = supported[l]
    print(f"Letter {l}: {r['correct']}/{r['total']} correct | NoMatch: {r['nomatch']} | Conflict: {r['conflict']} | Wrong: {r['wrong']}")

print(f"\nUnsupported letters: {unsupported['unknown']}/{unsupported['total']} returned UNKNOWN safely!")
print(f"False Positives: {unsupported['fp']}")
for fp in unsupported["fp_list"]:
    print(f"  {fp[0]} -> {fp[1]} (frame {fp[2]}, {fp[3]})")
