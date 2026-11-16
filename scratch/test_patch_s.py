import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np

with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace constants
old_constants = """# 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón - QIPEDC D0512)
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

new_constants = """# 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón - QIPEDC D0512 & Webcam)
S_D_TW_MAX = 1.40
S_D_TI_MIN = 0.10
S_D_TI_MAX = 0.48
S_D_TM_MIN = 0.35
S_D_TM_MAX = 0.78
S_D_IW_MAX = 1.35
S_D_MW_MAX = 0.85
S_D_RW_MAX = 0.75
S_D_PW_MIN = 0.35
S_D_PW_MAX = 0.85
S_THUMB_LAT_MIN = 0.55"""

old_rule = """        # 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón - QIPEDC D0512)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_tw <= S_D_TW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX and 
                S_D_TM_MIN <= d_tm <= S_D_TM_MAX and d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX and
                d_rw <= S_D_RW_MAX and d_pw <= S_D_PW_MAX and
                thumb_lat >= S_THUMB_LAT_MIN):"""

new_rule = """        # 9. Chữ S (Nắm đấm tròn, ngón cái vắt ngang mặt ngoài 4 ngón - QIPEDC D0512 & Webcam)
        # -------------------------------------------------------------------
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if (d_tw <= S_D_TW_MAX and S_D_TI_MIN <= d_ti <= S_D_TI_MAX and 
                S_D_TM_MIN <= d_tm <= S_D_TM_MAX and d_iw <= S_D_IW_MAX and d_mw <= S_D_MW_MAX and
                d_rw <= S_D_RW_MAX and S_D_PW_MIN <= d_pw <= S_D_PW_MAX and
                thumb_lat >= S_THUMB_LAT_MIN):"""

old_diag = """            "d_rw_le_max": (d_rw <= S_D_RW_MAX) if d_rw is not None else None,
            "d_pw_le_max": (d_pw <= S_D_PW_MAX) if d_pw is not None else None,
            "thumb_lat_ge_min": (thumb_lat >= S_THUMB_LAT_MIN) if thumb_lat is not None else None,"""

new_diag = """            "d_rw_le_max": (d_rw <= S_D_RW_MAX) if d_rw is not None else None,
            "d_pw_in_range": (S_D_PW_MIN <= d_pw <= S_D_PW_MAX) if d_pw is not None else None,
            "thumb_lat_ge_min": (thumb_lat >= S_THUMB_LAT_MIN) if thumb_lat is not None else None,"""

assert old_constants in text
assert old_rule in text
assert old_diag in text
print("All 3 targets found perfectly!")
