import json
import sys
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac

with open("scratch/d0509n_extracted_raw.json", encoding="utf-8") as f:
    samples_p = json.load(f)

for fidx in [24, 25, 26, 27, 28, 29, 30]:
    lm = samples_p[fidx]["landmarks"]
    res = ac.alphabet_classifier.classify_detailed(lm)
    feat = res.get("features", {})
    
    d_iw = feat.get("d_IW")
    d_mw = feat.get("d_MW")
    d_ti = feat.get("d_TI")
    d_tm = feat.get("d_TM")
    d_tw = feat.get("d_TW")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    ang_im_dir = feat.get("ang_IM_dir")
    idx_straight = feat.get("idx_straight")
    idx_tip_y = feat.get("idx_tip_y")
    idx_dy = feat.get("idx_dy")
    idx_horiz = feat.get("idx_horiz_ratio")
    mid_straight = feat.get("mid_straight")
    mid_tip_y = feat.get("mid_tip_y")

    conds_down = {
        "down_d_iw_ge_min": (d_iw, ">=", ac.P_DOWN_D_IW_MIN, d_iw >= ac.P_DOWN_D_IW_MIN),
        "down_idx_straight_ge_min": (idx_straight, ">=", ac.P_DOWN_IDX_STRAIGHT_MIN, idx_straight >= ac.P_DOWN_IDX_STRAIGHT_MIN),
        "down_idx_tip_y_ge_min": (idx_tip_y, ">=", ac.P_DOWN_IDX_TIP_Y_MIN, idx_tip_y >= ac.P_DOWN_IDX_TIP_Y_MIN),
        "down_idx_dy_ge_min": (idx_dy, ">=", ac.P_DOWN_IDX_DY_MIN, idx_dy >= ac.P_DOWN_IDX_DY_MIN),
        "down_idx_horiz_le_max": (idx_horiz, "<=", ac.P_DOWN_IDX_HORIZ_MAX, idx_horiz <= ac.P_DOWN_IDX_HORIZ_MAX),
        "down_d_mw_ge_min": (d_mw, ">=", ac.P_DOWN_D_MW_MIN, d_mw >= ac.P_DOWN_D_MW_MIN),
        "down_mid_straight_ge_min": (mid_straight, ">=", ac.P_DOWN_MID_STRAIGHT_MIN, mid_straight >= ac.P_DOWN_MID_STRAIGHT_MIN),
        "down_mid_tip_y_ge_min": (mid_tip_y, ">=", ac.P_DOWN_MID_TIP_Y_MIN, mid_tip_y >= ac.P_DOWN_MID_TIP_Y_MIN),
        "down_d_tm_le_max": (d_tm, "<=", ac.P_DOWN_D_TM_MAX, d_tm <= ac.P_DOWN_D_TM_MAX),
        "down_d_ti_le_max": (d_ti, "<=", ac.P_DOWN_D_TI_MAX, d_ti <= ac.P_DOWN_D_TI_MAX),
        "down_d_tw_ge_min": (d_tw, ">=", ac.P_DOWN_D_TW_MIN, d_tw >= ac.P_DOWN_D_TW_MIN),
        "down_ang_im_dir_in_range": (ang_im_dir, f"[{ac.P_DOWN_ANG_IM_MIN}, {ac.P_DOWN_ANG_IM_MAX}]", True, ac.P_DOWN_ANG_IM_MIN <= ang_im_dir <= ac.P_DOWN_ANG_IM_MAX),
        "down_d_rw_le_max": (d_rw, "<=", ac.P_DOWN_D_RW_MAX, d_rw <= ac.P_DOWN_D_RW_MAX),
        "down_d_pw_le_max": (d_pw, "<=", ac.P_DOWN_D_PW_MAX, d_pw <= ac.P_DOWN_D_PW_MAX),
    }
    failed = [f"{k}: val={v[0]:.2f} {v[1]} thresh={v[2]}" for k, v in conds_down.items() if not v[3]]
    print(f"Frame {samples_p[fidx]['frame_index']:2d}: Failed {len(failed)} conditions: {failed}")
