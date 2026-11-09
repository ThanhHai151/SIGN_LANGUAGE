import subprocess
import re

out = subprocess.check_output(['git', 'cat-file', '-p', '67357e36:backend/vision/alphabet_classifier.py'])
text = out.decode('utf-8')

# 1. Update T constants
target_t_const = """# 5. Chữ T (Ngón cái kẹp giữa trỏ và giữa, trỏ quặp trên ngón cái)
T_D_TI_MAX = 0.16
T_D_TM_MIN = 0.80
T_D_IM_MIN = 0.80
T_D_IW_MAX = 1.30"""
repl_t_const = """# 5. Chữ T (Ngón cái chụm đầu ngón trỏ hình chữ O, 3 ngón giữa/áp út/út duỗi thẳng đứng - QIPEDC D0513B)
T_D_TI_MAX = 0.20
T_D_TM_MIN = 0.70
T_D_IM_MIN = 0.75
T_D_IW_MAX = 1.30
T_D_MW_MIN = 1.65
T_D_RW_MIN = 1.55"""
assert target_t_const in text
text = text.replace(target_t_const, repl_t_const, 1)

# 2. Update O constants
target_o_const = """# 6. Chữ O (Vòng tròn kín chụm đầu các ngón vào ngón cái)
O_D_TI_MAX = 0.22
O_D_TM_MAX = 0.20
O_D_IW_MIN = 0.95
O_D_MW_MIN = 0.90"""
repl_o_const = """# 6. Chữ O (Vòng tròn kín chụm đầu các ngón vào ngón cái - QIPEDC D0506)
O_D_TI_MAX = 0.22
O_D_TM_MAX = 0.22
O_D_IW_MIN = 0.95
O_D_MW_MIN = 0.90
O_D_RW_MIN = 0.80
O_D_PW_MIN = 0.75
O_IDX_STRAIGHT_MAX = 0.40"""
assert target_o_const in text
text = text.replace(target_o_const, repl_o_const, 1)

# 3. Update S constants
target_s_const = """# 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón)
S_D_TW_MAX = 0.90
S_D_TI_MIN = 0.25
S_D_TI_MAX = 0.45
S_D_TM_MIN = 0.40
S_D_TM_MAX = 0.65
S_D_IW_MAX = 0.90
S_D_MW_MAX = 0.75"""
repl_s_const = """# 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón - QIPEDC D0512)
S_D_TW_MAX = 0.95
S_D_TI_MIN = 0.25
S_D_TI_MAX = 0.45
S_D_TM_MIN = 0.40
S_D_TM_MAX = 0.65
S_D_IW_MAX = 0.90
S_D_MW_MAX = 0.75
S_D_RW_MAX = 0.70
S_D_PW_MAX = 0.70
S_THUMB_LAT_MIN = 0.50"""
assert target_s_const in text
text = text.replace(target_s_const, repl_s_const, 1)

# 4. Add U, P, Q, R, X constants after K constants
target_k_const = """K_D_TI_MAX = 1.10
K_D_TM_MAX = 0.85
K_THUMB_MID_MAX = 0.50"""
repl_after_k = """K_D_TI_MAX = 1.10
K_D_TM_MAX = 0.85
K_THUMB_MID_MAX = 0.50

# 14. Chữ U (Trỏ & Giữa duỗi thẳng đứng song song khép sát, áp út & út gập - QIPEDC D0514 & D0515N)
U_D_IW_MIN = 1.65
U_D_MW_MIN = 1.65
U_D_RW_MAX = 1.05
U_D_PW_MAX = 1.05
U_D_IM_MAX = 0.22
U_ANG_IM_MAX = 6.5
U_DIFF_X_IM_MAX = 0.20
U_IDX_HORIZ_MAX = 0.35
U_IDX_TIP_Y_MAX = -0.80
U_THUMB_RING_MIN = 0.22

# 15. Chữ P (Ngón trỏ duỗi đứng, ngón giữa gập ngang, ngón cái tì ngón giữa - QIPEDC D0509B)
P_D_IW_MIN = 1.70
P_IDX_STRAIGHT_MIN = 0.95
P_IDX_TIP_Y_MAX = -1.60
P_IDX_HORIZ_MAX = 0.20
P_D_MW_MAX = 1.25
P_D_TI_MIN = 1.35
P_D_TM_MAX = 0.55
P_D_TW_MAX = 1.05
P_THUMB_LAT_MIN = 0.65
P_ANG_IM_MIN = 105.0

# 16. Chữ Q (Ngón trỏ và ngón cái duỗi chúc xuống đất, áp út & út gập - QIPEDC D0510)
Q_D_IW_MIN = 1.80
Q_D_IW_MAX = 2.25
Q_IDX_STRAIGHT_MIN = 0.90
Q_IDX_TIP_Y_MIN = 0.30
Q_IDX_TIP_Y_MAX = 1.25
Q_IDX_DY_MIN = 0.45
Q_IDX_HORIZ_MIN = 0.50
Q_D_TW_MIN = 1.25
Q_D_TW_MAX = 1.60
Q_D_TI_MIN = 0.50
Q_D_TI_MAX = 0.88
Q_ANG_TI_MAX = 15.0
Q_MID_STRAIGHT_MAX = 0.50
Q_M_CURL_MIN = 0.40
Q_ANG_IM_MIN = 30.0

# 17. Chữ R (Ngón giữa bắt chéo qua sau ngón trỏ, các ngón khác nắm - QIPEDC D0511)
R_D_IW_MIN = 1.10
R_D_MW_MIN = 1.25
R_D_RW_MAX = 0.65
R_D_PW_MAX = 0.55
R_D_IM_MAX = 0.55
R_IDX_STRAIGHT_MIN = 0.85
R_MID_STRAIGHT_MIN = 0.85
R_IDX_TIP_Y_MAX = -0.80
R_MID_TIP_Y_MAX = -1.00
R_CROSS_IND_MAX = 0.0050
R_D_MW_IW_DIFF_MIN = -0.25

# 18. Chữ X (Ngón trỏ cong móc chạm ngón cái, các ngón khác nắm - QIPEDC D0517B)
X_D_TI_MAX = 0.25
X_D_IW_MAX = 1.10
X_IDX_CURL_MIN = 0.15
X_IDX_STRAIGHT_MAX = 0.85
X_D_RW_MAX = 0.70
X_D_PW_MAX = 0.60
X_D_TW_MIN = 0.70
X_D_TW_MAX = 0.95
X_IDX_TIP_Y_MAX = -0.20"""
assert target_k_const in text
text = text.replace(target_k_const, repl_after_k, 1)

# 5. Update extract_features with idx_dx, idx_dy and cross_indicator
target_extract = """        # Đặc trưng định hướng bàn tay và quan hệ ngón cái để phân biệt H (Miền Nam) vs V
        mcp_width_2d = float(math.hypot(sh[17][0] - sh[5][0], sh[17][1] - sh[5][1]) / palm_scale)
        thumb_ring_dist = float(calculate_distance(sh[4], sh[14]) / palm_scale)
        thumb_mid_dist = float(calculate_distance(sh[4], sh[10]) / palm_scale)

        return {
            "palm_scale": round(palm_scale, 4),
            "thumb_ext": t_ext,
            "index_ext": i_ext,
            "middle_ext": m_ext,
            "ring_ext": r_ext,
            "pinky_ext": p_ext,
            "index_curl": round(i_curl, 4),
            "middle_curl": round(m_curl, 4),
            "ring_curl": round(r_curl, 4),
            "pinky_curl": round(p_curl, 4),
            "ang_TI": round(ang_ti, 2),
            "ang_TP": round(ang_tp, 2),
            "thumb_lat": round(thumb_lat, 4),
            "d_TI": round(d_ti, 4),
            "d_TM": round(d_tm, 4),
            "d_TW": round(d_tw, 4),
            "d_IM": round(d_im, 4),
            "d_IW": round(d_iw, 4),
            "d_MW": round(d_mw, 4),
            "d_RW": round(d_rw, 4),
            "d_PW": round(d_pw, 4),
            "diff_x_IM": round(diff_x_im, 4),
            "idx_horiz_ratio": round(idx_horiz_ratio, 4),
            "idx_straight": round(idx_straight, 4),
            "idx_tip_y": round(idx_tip_y, 4),
            "idx_dx": round(vx_58, 4),
            "mid_horiz_ratio": round(mid_horiz_ratio, 4),
            "mid_straight": round(mid_straight, 4),
            "mid_tip_y": round(mid_tip_y, 4),
            "ang_IM_dir": round(ang_im_dir, 2),
            "mcp_width_2d": round(mcp_width_2d, 4),
            "thumb_ring_dist": round(thumb_ring_dist, 4),
            "thumb_mid_dist": round(thumb_mid_dist, 4),
        }"""
repl_extract = """        # Đặc trưng định hướng bàn tay và quan hệ ngón cái để phân biệt H (Miền Nam) vs V
        mcp_width_2d = float(math.hypot(sh[17][0] - sh[5][0], sh[17][1] - sh[5][1]) / palm_scale)
        thumb_ring_dist = float(calculate_distance(sh[4], sh[14]) / palm_scale)
        thumb_mid_dist = float(calculate_distance(sh[4], sh[10]) / palm_scale)

        # Đặc trưng bắt chéo ngón tay (ngón giữa bắt chéo qua ngón trỏ cho chữ R)
        base_dx = float((sh[9][0] - sh[5][0]) / palm_scale)
        tip_dx = float((sh[12][0] - sh[8][0]) / palm_scale)
        cross_indicator = float(base_dx * tip_dx)

        return {
            "palm_scale": round(palm_scale, 4),
            "thumb_ext": t_ext,
            "index_ext": i_ext,
            "middle_ext": m_ext,
            "ring_ext": r_ext,
            "pinky_ext": p_ext,
            "index_curl": round(i_curl, 4),
            "middle_curl": round(m_curl, 4),
            "ring_curl": round(r_curl, 4),
            "pinky_curl": round(p_curl, 4),
            "ang_TI": round(ang_ti, 2),
            "ang_TP": round(ang_tp, 2),
            "thumb_lat": round(thumb_lat, 4),
            "d_TI": round(d_ti, 4),
            "d_TM": round(d_tm, 4),
            "d_TW": round(d_tw, 4),
            "d_IM": round(d_im, 4),
            "d_IW": round(d_iw, 4),
            "d_MW": round(d_mw, 4),
            "d_RW": round(d_rw, 4),
            "d_PW": round(d_pw, 4),
            "diff_x_IM": round(diff_x_im, 4),
            "idx_horiz_ratio": round(idx_horiz_ratio, 4),
            "idx_straight": round(idx_straight, 4),
            "idx_tip_y": round(idx_tip_y, 4),
            "idx_dx": round(vx_58, 4),
            "idx_dy": round(vy_58, 4),
            "mid_horiz_ratio": round(mid_horiz_ratio, 4),
            "mid_straight": round(mid_straight, 4),
            "mid_tip_y": round(mid_tip_y, 4),
            "ang_IM_dir": round(ang_im_dir, 2),
            "mcp_width_2d": round(mcp_width_2d, 4),
            "thumb_ring_dist": round(thumb_ring_dist, 4),
            "thumb_mid_dist": round(thumb_mid_dist, 4),
            "cross_indicator": round(cross_indicator, 6),
        }"""
assert target_extract in text
text = text.replace(target_extract, repl_extract, 1)

# 6. Update evaluate_rules for T
target_t_eval = """        # -------------------------------------------------------------------
        # 5. Chữ T (Ngón cái kẹp giữa ngón trỏ và ngón giữa)
        # -------------------------------------------------------------------
        if d_ti <= T_D_TI_MAX and d_tm >= T_D_TM_MIN and d_im >= T_D_IM_MIN and d_iw <= T_D_IW_MAX:
            matches.append(("T", f"Ngón cái kẹp giữa trỏ & giữa: d_TI={d_ti:.2f}<={T_D_TI_MAX}, d_IM={d_im:.2f}>={T_D_IM_MIN}", 0.95))"""
repl_t_eval = """        # -------------------------------------------------------------------
        # 5. Chữ T (Ngón cái chụm đầu ngón trỏ hình chữ O, 3 ngón giữa/áp út/út duỗi thẳng đứng - QIPEDC D0513B)
        # -------------------------------------------------------------------
        if (d_ti <= T_D_TI_MAX and d_tm >= T_D_TM_MIN and d_im >= T_D_IM_MIN and
            d_iw <= T_D_IW_MAX and d_mw >= T_D_MW_MIN and d_rw >= T_D_RW_MIN):
            matches.append(("T", f"Ngón cái chụm đầu ngón trỏ, giữa/áp út/út duỗi (T chuẩn VSL): d_TI={d_ti:.2f}<={T_D_TI_MAX}, d_IM={d_im:.2f}>={T_D_IM_MIN}, d_MW={d_mw:.2f}>={T_D_MW_MIN}", 0.95))"""
assert target_t_eval in text
text = text.replace(target_t_eval, repl_t_eval, 1)

# 7. Update evaluate_rules for O
target_o_eval = """        # -------------------------------------------------------------------
        # 6. Chữ O (Vòng tròn kín chụm đầu các ngón vào ngón cái)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if d_ti <= O_D_TI_MAX and d_tm <= O_D_TM_MAX and d_iw >= O_D_IW_MIN and d_mw >= O_D_MW_MIN:
                matches.append(("O", f"Vòng tròn kín chữ O: d_TI={d_ti:.2f}<={O_D_TI_MAX}, d_TM={d_tm:.2f}<={O_D_TM_MAX}", 0.95))"""
repl_o_eval = """        # -------------------------------------------------------------------
        # 6. Chữ O (Vòng tròn kín chụm đầu các ngón vào ngón cái - QIPEDC D0506)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_ti <= O_D_TI_MAX and d_tm <= O_D_TM_MAX and d_iw >= O_D_IW_MIN and d_mw >= O_D_MW_MIN and
                d_rw >= O_D_RW_MIN and d_pw >= O_D_PW_MIN and idx_straight <= O_IDX_STRAIGHT_MAX):
                matches.append(("O", f"Vòng tròn kín chữ O: d_TI={d_ti:.2f}<={O_D_TI_MAX}, d_TM={d_tm:.2f}<={O_D_TM_MAX}, d_RW={d_rw:.2f}>={O_D_RW_MIN}", 0.95))"""
assert target_o_eval in text
text = text.replace(target_o_eval, repl_o_eval, 1)

# 8. Update evaluate_rules for S and E
target_s_eval = """        # -------------------------------------------------------------------
        # 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_tw <= S_D_TW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX and 
                S_D_TM_MIN <= d_tm <= S_D_TM_MAX and d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX):
                matches.append(("S", f"Nắm đấm ngón cái vắt ngang: d_TI={d_ti:.2f}, d_TW={d_tw:.2f}<={S_D_TW_MAX}", 0.95))

        # -------------------------------------------------------------------
        # 10. Chữ E (4 ngón tì gập lên ngón cái)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and (not p_ext or d_pw <= E_D_PW_MAX):
            if (d_tw <= E_D_TW_MAX and 
                d_ti <= E_D_TI_MAX and 
                E_D_TM_MIN <= d_tm <= E_D_TM_MAX and 
                E_D_PW_MIN <= d_pw <= E_D_PW_MAX):
                # Loại trừ chữ S (nắm đấm ngón cái vắt ngang với d_iw <= 0.90, d_mw <= 0.75)
                is_s_fist = (d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX)
                if not is_s_fist:
                    matches.append(("E", f"4 ngón tì gập lên ngón cái: d_TM={d_tm:.2f}, d_TW={d_tw:.2f}<={E_D_TW_MAX}", 0.95))"""
repl_s_eval = """        # -------------------------------------------------------------------
        # 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón - QIPEDC D0512)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_tw <= S_D_TW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX and 
                S_D_TM_MIN <= d_tm <= S_D_TM_MAX and d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX and
                d_rw <= S_D_RW_MAX and d_pw <= S_D_PW_MAX and
                thumb_lat >= S_THUMB_LAT_MIN):
                matches.append(("S", f"Nắm đấm ngón cái vắt ngang: d_TI={d_ti:.2f}, d_TW={d_tw:.2f}<={S_D_TW_MAX}, thumb_lat={thumb_lat:.2f}>={S_THUMB_LAT_MIN}", 0.95))

        # -------------------------------------------------------------------
        # 10. Chữ E (4 ngón tì gập lên ngón cái)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and (not p_ext or d_pw <= E_D_PW_MAX):
            if (d_tw <= E_D_TW_MAX and 
                d_ti <= E_D_TI_MAX and 
                E_D_TM_MIN <= d_tm <= E_D_TM_MAX and 
                E_D_PW_MIN <= d_pw <= E_D_PW_MAX):
                # Loại trừ chữ S (nắm đấm ngón cái vắt ngang với d_iw <= 0.90, d_mw <= 0.75, thumb_lat >= 0.50)
                is_s_fist = (d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX and thumb_lat >= S_THUMB_LAT_MIN)
                if not is_s_fist:
                    matches.append(("E", f"4 ngón tì gập lên ngón cái: d_TM={d_tm:.2f}, d_TW={d_tw:.2f}<={E_D_TW_MAX}", 0.95))"""
assert target_s_eval in text
text = text.replace(target_s_eval, repl_s_eval, 1)

# 9. Add U, P, Q, R, X to evaluate_rules at the end
target_end_eval = """        if is_k_index and is_k_middle and is_k_angle and is_k_curled and is_k_thumb:
            # Loại bỏ các phán đoán nhầm V, D, L, G, H nếu có
            matches[:] = [m for m in matches if m[0] not in ("V", "D", "L", "G", "H")]
            matches.append((
                "K",
                f"Chữ K: trỏ đứng d_IW={d_iw:.2f}, giữa ngang d_MW={d_mw:.2f}, ang_IM={ang_im_dir:.1f}°",
                0.95
            ))

        return matches"""
repl_end_eval = """        if is_k_index and is_k_middle and is_k_folded and is_k_spread and is_k_thumb:
            matches.append(("K", f"Chữ K (trỏ đứng, giữa ngang, cái kẹp): d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, ang_IM={ang_im_dir:.1f}°", 0.95))

        # -------------------------------------------------------------------
        # 14. Chữ U (Trỏ & Giữa duỗi thẳng đứng song song khép sát, áp út & út gập - QIPEDC D0514 & D0515N)
        # -------------------------------------------------------------------
        is_u_fingers = (
            d_iw >= U_D_IW_MIN
            and d_mw >= U_D_MW_MIN
            and d_rw <= U_D_RW_MAX
            and d_pw <= U_D_PW_MAX
            and (not r_ext)
            and (not p_ext)
        )
        is_u_parallel = (
            d_im <= U_D_IM_MAX
            and ang_im_dir <= U_ANG_IM_MAX
            and diff_x_im <= U_DIFF_X_IM_MAX
        )
        is_u_upright = (
            idx_horiz_ratio <= U_IDX_HORIZ_MAX
            and idx_tip_y <= U_IDX_TIP_Y_MAX
        )

        if is_u_fingers and is_u_parallel and is_u_upright:
            # Loại bỏ các phán đoán nhầm V, D, L, K nếu có
            matches[:] = [m for m in matches if m[0] not in ("V", "D", "L", "K")]
            matches.append((
                "U",
                f"Chữ U (trỏ & giữa duỗi song song khép sát): d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_IM={d_im:.2f}<={U_D_IM_MAX}, ang_IM={ang_im_dir:.1f}°<={U_ANG_IM_MAX}°",
                0.95
            ))

        # -------------------------------------------------------------------
        # 15. Chữ P (Ngón trỏ duỗi đứng, ngón giữa gập ngang, ngón cái tì ngón giữa - QIPEDC D0509B)
        # -------------------------------------------------------------------
        is_p_index = (
            d_iw >= P_D_IW_MIN
            and idx_straight >= P_IDX_STRAIGHT_MIN
            and idx_tip_y <= P_IDX_TIP_Y_MAX
            and idx_horiz_ratio <= P_IDX_HORIZ_MAX
        )
        is_p_middle = (
            d_mw <= P_D_MW_MAX
        )
        is_p_thumb = (
            d_ti >= P_D_TI_MIN
            and d_tm <= P_D_TM_MAX
            and d_tw <= P_D_TW_MAX
            and thumb_lat >= P_THUMB_LAT_MIN
        )
        is_p_angle = (
            ang_im_dir >= P_ANG_IM_MIN
        )

        if is_p_index and is_p_middle and is_p_thumb and is_p_angle:
            # Loại bỏ các phán đoán nhầm K, V, D, L nếu có
            matches[:] = [m for m in matches if m[0] not in ("K", "V", "D", "L")]
            matches.append((
                "P",
                f"Chữ P: trỏ đứng d_IW={d_iw:.2f}, cái tì giữa d_TM={d_tm:.2f}<={P_D_TM_MAX}, ang_IM={ang_im_dir:.1f}°>={P_ANG_IM_MIN}°",
                0.95
            ))

        # -------------------------------------------------------------------
        # 16. Chữ Q (Ngón trỏ và ngón cái duỗi chúc xuống đất, áp út & út gập - QIPEDC D0510)
        # -------------------------------------------------------------------
        idx_dy = float(feat.get("idx_dy", 0.0))
        is_q_index = (
            Q_D_IW_MIN <= d_iw <= Q_D_IW_MAX
            and idx_straight >= Q_IDX_STRAIGHT_MIN
            and Q_IDX_TIP_Y_MIN <= idx_tip_y <= Q_IDX_TIP_Y_MAX
            and idx_dy >= Q_IDX_DY_MIN
            and idx_horiz_ratio >= Q_IDX_HORIZ_MIN
        )
        is_q_thumb = (
            Q_D_TW_MIN <= d_tw <= Q_D_TW_MAX
            and Q_D_TI_MIN <= d_ti <= Q_D_TI_MAX
            and ang_ti <= Q_ANG_TI_MAX
        )
        is_q_curled = (
            mid_straight <= Q_MID_STRAIGHT_MAX
            and m_curl >= Q_M_CURL_MIN
            and ang_im_dir >= Q_ANG_IM_MIN
        )

        if is_q_index and is_q_thumb and is_q_curled:
            # Loại bỏ các phán đoán nhầm G, D, L nếu có
            matches[:] = [m for m in matches if m[0] not in ("G", "D", "L")]
            matches.append((
                "Q",
                f"Chữ Q: trỏ & cái chúc xuống đất d_IW={d_iw:.2f}, idx_dy={idx_dy:.2f}>={Q_IDX_DY_MIN}, d_TW={d_tw:.2f}>={Q_D_TW_MIN}",
                0.95
            ))

        # -------------------------------------------------------------------
        # 17. Chữ R (Ngón giữa bắt chéo qua sau ngón trỏ, các ngón khác nắm - QIPEDC D0511)
        # -------------------------------------------------------------------
        cross_ind = float(feat.get("cross_indicator", 1.0))
        is_r_fingers = (
            d_iw >= R_D_IW_MIN
            and d_mw >= R_D_MW_MIN
            and d_rw <= R_D_RW_MAX
            and d_pw <= R_D_PW_MAX
            and idx_straight >= R_IDX_STRAIGHT_MIN
            and mid_straight >= R_MID_STRAIGHT_MIN
        )
        is_r_upright = (
            idx_tip_y <= R_IDX_TIP_Y_MAX
            and mid_tip_y <= R_MID_TIP_Y_MAX
        )
        is_r_cross = (
            d_im <= R_D_IM_MAX
            and cross_ind <= R_CROSS_IND_MAX
            and (d_mw - d_iw) >= R_D_MW_IW_DIFF_MIN
        )

        if is_r_fingers and is_r_upright and is_r_cross:
            # Loại bỏ các phán đoán nhầm U, V, K, D nếu có
            matches[:] = [m for m in matches if m[0] not in ("U", "V", "K", "D")]
            matches.append((
                "R",
                f"Chữ R (trỏ & giữa bắt chéo): d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, cross_ind={cross_ind:.4f}<={R_CROSS_IND_MAX}",
                0.95
            ))

        # -------------------------------------------------------------------
        # 18. Chữ X (Ngón trỏ cong móc chạm ngón cái, các ngón khác nắm - QIPEDC D0517B)
        # -------------------------------------------------------------------
        is_x_hook = (
            d_ti <= X_D_TI_MAX
            and d_iw <= X_D_IW_MAX
            and i_curl >= X_IDX_CURL_MIN
            and idx_straight <= X_IDX_STRAIGHT_MAX
        )
        is_x_curled = (
            d_rw <= X_D_RW_MAX
            and d_pw <= X_D_PW_MAX
        )
        is_x_thumb = (
            X_D_TW_MIN <= d_tw <= X_D_TW_MAX
        )
        is_x_upright = (
            idx_tip_y <= X_IDX_TIP_Y_MAX
        )

        if is_x_hook and is_x_curled and is_x_thumb and is_x_upright:
            # Loại bỏ các phán đoán nhầm A, S, O, E, C nếu có
            matches[:] = [m for m in matches if m[0] not in ("A", "S", "O", "E", "C")]
            matches.append((
                "X",
                f"Chữ X (trỏ cong móc chạm cái): d_TI={d_ti:.2f}<={X_D_TI_MAX}, d_TW={d_tw:.2f}, idx_curl={i_curl:.2f}>={X_IDX_CURL_MIN}",
                0.95
            ))

        return matches"""
assert target_end_eval in text
text = text.replace(target_end_eval, repl_end_eval, 1)

# 10. Add diagnostics methods for T, U, O, S, P, Q, R, X before classify_detailed
diag_methods = '''    def evaluate_rule_t_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule T."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_ti = _num("d_TI")
        d_tm = _num("d_TM")
        d_im = _num("d_IM")
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        conds = {
            "d_ti_le_max": (d_ti <= T_D_TI_MAX) if d_ti is not None else None,
            "d_tm_ge_min": (d_tm >= T_D_TM_MIN) if d_tm is not None else None,
            "d_im_ge_min": (d_im >= T_D_IM_MIN) if d_im is not None else None,
            "d_iw_le_max": (d_iw <= T_D_IW_MAX) if d_iw is not None else None,
            "d_mw_ge_min": (d_mw >= T_D_MW_MIN) if d_mw is not None else None,
            "d_rw_ge_min": (d_rw >= T_D_RW_MIN) if d_rw is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_u_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule U."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        d_im = _num("d_IM")
        ang_im_dir = _num("ang_IM_dir")
        diff_x_im = _num("diff_x_IM")
        idx_horiz = _num("idx_horiz_ratio")
        idx_tip_y = _num("idx_tip_y")
        conds = {
            "d_iw_ge_min": (d_iw >= U_D_IW_MIN) if d_iw is not None else None,
            "d_mw_ge_min": (d_mw >= U_D_MW_MIN) if d_mw is not None else None,
            "d_rw_le_max": (d_rw <= U_D_RW_MAX) if d_rw is not None else None,
            "d_pw_le_max": (d_pw <= U_D_PW_MAX) if d_pw is not None else None,
            "d_im_le_max": (d_im <= U_D_IM_MAX) if d_im is not None else None,
            "ang_im_dir_le_max": (ang_im_dir <= U_ANG_IM_MAX) if ang_im_dir is not None else None,
            "diff_x_im_le_max": (diff_x_im <= U_DIFF_X_IM_MAX) if diff_x_im is not None else None,
            "idx_horiz_le_max": (idx_horiz <= U_IDX_HORIZ_MAX) if idx_horiz is not None else None,
            "idx_tip_y_le_max": (idx_tip_y <= U_IDX_TIP_Y_MAX) if idx_tip_y is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_o_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule O."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_ti = _num("d_TI")
        d_tm = _num("d_TM")
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        idx_straight = _num("idx_straight")
        conds = {
            "d_ti_le_max": (d_ti <= O_D_TI_MAX) if d_ti is not None else None,
            "d_tm_le_max": (d_tm <= O_D_TM_MAX) if d_tm is not None else None,
            "d_iw_ge_min": (d_iw >= O_D_IW_MIN) if d_iw is not None else None,
            "d_mw_ge_min": (d_mw >= O_D_MW_MIN) if d_mw is not None else None,
            "d_rw_ge_min": (d_rw >= O_D_RW_MIN) if d_rw is not None else None,
            "d_pw_ge_min": (d_pw >= O_D_PW_MIN) if d_pw is not None else None,
            "idx_straight_le_max": (idx_straight <= O_IDX_STRAIGHT_MAX) if idx_straight is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_s_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule S."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_tw = _num("d_TW")
        d_ti = _num("d_TI")
        d_tm = _num("d_TM")
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        thumb_lat = _num("thumb_lat")
        conds = {
            "d_tw_le_max": (d_tw <= S_D_TW_MAX) if d_tw is not None else None,
            "d_ti_in_range": (S_D_TI_MIN <= d_ti <= S_D_TI_MAX) if d_ti is not None else None,
            "d_tm_in_range": (S_D_TM_MIN <= d_tm <= S_D_TM_MAX) if d_tm is not None else None,
            "d_iw_le_max": (d_iw <= S_D_IW_MAX) if d_iw is not None else None,
            "d_mw_le_max": (d_mw <= S_D_MW_MAX) if d_mw is not None else None,
            "d_rw_le_max": (d_rw <= S_D_RW_MAX) if d_rw is not None else None,
            "d_pw_le_max": (d_pw <= S_D_PW_MAX) if d_pw is not None else None,
            "thumb_lat_ge_min": (thumb_lat >= S_THUMB_LAT_MIN) if thumb_lat is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_p_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule P."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_ti = _num("d_TI")
        d_tm = _num("d_TM")
        d_tw = _num("d_TW")
        thumb_lat = _num("thumb_lat")
        ang_im_dir = _num("ang_IM_dir")
        idx_straight = _num("idx_straight")
        idx_tip_y = _num("idx_tip_y")
        idx_horiz = _num("idx_horiz_ratio")
        conds = {
            "d_iw_ge_min": (d_iw >= P_D_IW_MIN) if d_iw is not None else None,
            "idx_straight_ge_min": (idx_straight >= P_IDX_STRAIGHT_MIN) if idx_straight is not None else None,
            "idx_tip_y_le_max": (idx_tip_y <= P_IDX_TIP_Y_MAX) if idx_tip_y is not None else None,
            "idx_horiz_le_max": (idx_horiz <= P_IDX_HORIZ_MAX) if idx_horiz is not None else None,
            "d_mw_le_max": (d_mw <= P_D_MW_MAX) if d_mw is not None else None,
            "d_ti_ge_min": (d_ti >= P_D_TI_MIN) if d_ti is not None else None,
            "d_tm_le_max": (d_tm <= P_D_TM_MAX) if d_tm is not None else None,
            "d_tw_le_max": (d_tw <= P_D_TW_MAX) if d_tw is not None else None,
            "thumb_lat_ge_min": (thumb_lat >= P_THUMB_LAT_MIN) if thumb_lat is not None else None,
            "ang_im_dir_ge_min": (ang_im_dir >= P_ANG_IM_MIN) if ang_im_dir is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_q_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule Q."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_iw = _num("d_IW")
        d_tw = _num("d_TW")
        d_ti = _num("d_TI")
        ang_ti = _num("ang_TI")
        idx_straight = _num("idx_straight")
        idx_tip_y = _num("idx_tip_y")
        idx_dy = _num("idx_dy")
        idx_horiz = _num("idx_horiz_ratio")
        mid_straight = _num("mid_straight")
        m_curl = _num("middle_curl")
        ang_im_dir = _num("ang_IM_dir")
        conds = {
            "d_iw_in_range": (Q_D_IW_MIN <= d_iw <= Q_D_IW_MAX) if d_iw is not None else None,
            "idx_straight_ge_min": (idx_straight >= Q_IDX_STRAIGHT_MIN) if idx_straight is not None else None,
            "idx_tip_y_in_range": (Q_IDX_TIP_Y_MIN <= idx_tip_y <= Q_IDX_TIP_Y_MAX) if idx_tip_y is not None else None,
            "idx_dy_ge_min": (idx_dy >= Q_IDX_DY_MIN) if idx_dy is not None else None,
            "idx_horiz_ge_min": (idx_horiz >= Q_IDX_HORIZ_MIN) if idx_horiz is not None else None,
            "d_tw_in_range": (Q_D_TW_MIN <= d_tw <= Q_D_TW_MAX) if d_tw is not None else None,
            "d_ti_in_range": (Q_D_TI_MIN <= d_ti <= Q_D_TI_MAX) if d_ti is not None else None,
            "ang_ti_le_max": (ang_ti <= Q_ANG_TI_MAX) if ang_ti is not None else None,
            "mid_straight_le_max": (mid_straight <= Q_MID_STRAIGHT_MAX) if mid_straight is not None else None,
            "middle_curl_ge_min": (m_curl >= Q_M_CURL_MIN) if m_curl is not None else None,
            "ang_im_dir_ge_min": (ang_im_dir >= Q_ANG_IM_MIN) if ang_im_dir is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_r_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule R."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_iw = _num("d_IW")
        d_mw = _num("d_MW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        d_im = _num("d_IM")
        idx_straight = _num("idx_straight")
        mid_straight = _num("mid_straight")
        idx_tip_y = _num("idx_tip_y")
        mid_tip_y = _num("mid_tip_y")
        cross_ind = _num("cross_indicator")
        diff_mw_iw = ((d_mw - d_iw) >= R_D_MW_IW_DIFF_MIN) if (d_mw is not None and d_iw is not None) else None
        conds = {
            "d_iw_ge_min": (d_iw >= R_D_IW_MIN) if d_iw is not None else None,
            "d_mw_ge_min": (d_mw >= R_D_MW_MIN) if d_mw is not None else None,
            "d_rw_le_max": (d_rw <= R_D_RW_MAX) if d_rw is not None else None,
            "d_pw_le_max": (d_pw <= R_D_PW_MAX) if d_pw is not None else None,
            "idx_straight_ge_min": (idx_straight >= R_IDX_STRAIGHT_MIN) if idx_straight is not None else None,
            "mid_straight_ge_min": (mid_straight >= R_MID_STRAIGHT_MIN) if mid_straight is not None else None,
            "idx_tip_y_le_max": (idx_tip_y <= R_IDX_TIP_Y_MAX) if idx_tip_y is not None else None,
            "mid_tip_y_le_max": (mid_tip_y <= R_MID_TIP_Y_MAX) if mid_tip_y is not None else None,
            "d_im_le_max": (d_im <= R_D_IM_MAX) if d_im is not None else None,
            "cross_indicator_le_max": (cross_ind <= R_CROSS_IND_MAX) if cross_ind is not None else None,
            "diff_mw_iw_ge_min": diff_mw_iw,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}

    def evaluate_rule_x_diagnostics(self, feat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ghi nhận chẩn đoán Rule X."""
        if feat is None or not isinstance(feat, dict):
            return None
        def _num(key: str) -> Optional[float]:
            val = feat.get(key)
            if val is None or isinstance(val, bool): return None
            try:
                fv = float(val)
                return None if (math.isnan(fv) or math.isinf(fv)) else fv
            except (ValueError, TypeError): return None
        d_ti = _num("d_TI")
        d_iw = _num("d_IW")
        d_tw = _num("d_TW")
        d_rw = _num("d_RW")
        d_pw = _num("d_PW")
        idx_curl = _num("index_curl")
        idx_straight = _num("idx_straight")
        idx_tip_y = _num("idx_tip_y")
        conds = {
            "d_ti_le_max": (d_ti <= X_D_TI_MAX) if d_ti is not None else None,
            "d_iw_le_max": (d_iw <= X_D_IW_MAX) if d_iw is not None else None,
            "idx_curl_ge_min": (idx_curl >= X_IDX_CURL_MIN) if idx_curl is not None else None,
            "idx_straight_le_max": (idx_straight <= X_IDX_STRAIGHT_MAX) if idx_straight is not None else None,
            "d_rw_le_max": (d_rw <= X_D_RW_MAX) if d_rw is not None else None,
            "d_pw_le_max": (d_pw <= X_D_PW_MAX) if d_pw is not None else None,
            "d_tw_in_range": (X_D_TW_MIN <= d_tw <= X_D_TW_MAX) if d_tw is not None else None,
            "idx_tip_y_le_max": (idx_tip_y <= X_IDX_TIP_Y_MAX) if idx_tip_y is not None else None,
        }
        matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else None
        return {"matched": matched, "conditions": conds, "failed_conditions": [k for k, v in conds.items() if v is False]}
'''

target_class_detailed = "    def classify_detailed(self, landmarks: Any) -> Dict[str, Any]:"
assert target_class_detailed in text
text = text.replace(target_class_detailed, diag_methods + "\n" + target_class_detailed, 1)

with open('backend/vision/alphabet_classifier.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully written complete unified alphabet_classifier.py!")

