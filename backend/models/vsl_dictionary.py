"""
Từ điển Ngôn ngữ Ký hiệu Việt Nam (VSL Dictionary)
Bao gồm Bảng chữ cái ngón tay (Fingerspelling) và Từ vựng giao tiếp thông dụng (Core Lexicon).
Chứa siêu dữ liệu: từ ngữ, phiên âm, mô tả động tác, hướng bàn tay, và các vector tư thế 3D chuẩn.
"""

VSL_ALPHABET = {
    "A": {
        "label": "A",
        "type": "alphabet",
        "description": "Nắm cả bàn tay lại, ngón cái áp sát vào cạnh ngoài ngón trỏ, lòng bàn tay hướng về phía trước.",
        "fingers_state": {"thumb": "side", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "B": {
        "label": "B",
        "type": "alphabet",
        "description": "Bốn ngón trỏ, giữa, áp út và út duỗi thẳng khép sát nhau, ngón cái gập ngang qua lòng bàn tay.",
        "fingers_state": {"thumb": "folded", "index": "straight", "middle": "straight", "ring": "straight", "pinky": "straight"},
        "wrist_orientation": "up"
    },
    "C": {
        "label": "C",
        "type": "alphabet",
        "description": "Các ngón tay uốn cong nhẹ tạo thành hình chữ C hướng về phía người đối diện.",
        "fingers_state": {"thumb": "curved", "index": "curved", "middle": "curved", "ring": "curved", "pinky": "curved"},
        "wrist_orientation": "up"
    },
    "D": {
        "label": "D",
        "type": "alphabet",
        "description": "Ngón trỏ duỗi thẳng chỉ lên trên, ngón cái chạm đầu ngón giữa, áp út và út tạo thành vòng tròn khép kín.",
        "fingers_state": {"thumb": "touch_middle", "index": "straight", "middle": "circle", "ring": "circle", "pinky": "circle"},
        "wrist_orientation": "up"
    },
    "Đ": {
        "label": "Đ",
        "type": "alphabet",
        "description": "Tương tự chữ D nhưng ngón trỏ khẽ gập ngang hoặc có cử động gạch ngang biểu thị dấu gạch.",
        "fingers_state": {"thumb": "touch_middle", "index": "hooked", "middle": "circle", "ring": "circle", "pinky": "circle"},
        "wrist_orientation": "up"
    },
    "E": {
        "label": "E",
        "type": "alphabet",
        "description": "Năm đầu ngón tay khép cong lại, đầu ngón cái chạm nhẹ vào các đầu ngón tay gập.",
        "fingers_state": {"thumb": "under", "index": "bent", "middle": "bent", "ring": "bent", "pinky": "bent"},
        "wrist_orientation": "up"
    },
    "F": {
        "label": "F",
        "type": "alphabet",
        "description": "Đầu ngón cái chạm đầu ngón trỏ tạo thành vòng tròn khép kín (dấu OK), 3 ngón giữa, áp út và út duỗi thẳng xòe nhẹ.",
        "fingers_state": {"thumb": "touch_index", "index": "touch_thumb", "middle": "straight", "ring": "straight", "pinky": "straight"},
        "wrist_orientation": "up"
    },
    "G": {
        "label": "G",
        "type": "alphabet",
        "description": "Ngón cái và ngón trỏ duỗi song song hướng ngang sang trái/phải, các ngón còn lại gập vào lòng bàn tay.",
        "fingers_state": {"thumb": "straight_horizontal", "index": "straight_horizontal", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "side"
    },
    "H": {
        "label": "H",
        "type": "alphabet",
        "description": "Ngón trỏ và ngón giữa duỗi thẳng khép sát nhau hướng ngang, ngón cái giữ các ngón còn lại.",
        "fingers_state": {"thumb": "folded", "index": "horizontal", "middle": "horizontal", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "side"
    },
    "I": {
        "label": "I",
        "type": "alphabet",
        "description": "Chỉ ngón út duỗi thẳng đứng lên trên, bốn ngón còn lại nắm chặt vào lòng bàn tay.",
        "fingers_state": {"thumb": "folded", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "straight"},
        "wrist_orientation": "up"
    },
    "J": {
        "label": "J",
        "type": "alphabet",
        "description": "Ngón út duỗi thẳng vẽ nét móc chữ J trong không gian.",
        "fingers_state": {"thumb": "folded", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "straight_trace"},
        "wrist_orientation": "up"
    },
    "K": {
        "label": "K",
        "type": "alphabet",
        "description": "Ngón trỏ duỗi thẳng, ngón giữa duỗi nghiêng, đầu ngón cái tì vào giữa hai ngón trỏ và giữa.",
        "fingers_state": {"thumb": "between", "index": "straight", "middle": "angled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "L": {
        "label": "L",
        "type": "alphabet",
        "description": "Ngón trỏ và ngón cái duỗi vuông góc 90 độ tạo thành hình chữ L rõ nét.",
        "fingers_state": {"thumb": "open_90", "index": "straight", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "M": {
        "label": "M",
        "type": "alphabet",
        "description": "Ba ngón trỏ, giữa và áp út gập phủ qua ngón cái (ngón cái luồn thò ra dưới ngón áp út).",
        "fingers_state": {"thumb": "under_ring", "index": "folded_over", "middle": "folded_over", "ring": "folded_over", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "N": {
        "label": "N",
        "type": "alphabet",
        "description": "Hai ngón trỏ và giữa gập phủ qua ngón cái (ngón cái luồn thò ra dưới ngón giữa).",
        "fingers_state": {"thumb": "under_middle", "index": "folded_over", "middle": "folded_over", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "O": {
        "label": "O",
        "type": "alphabet",
        "description": "Tất cả các đầu ngón tay chụm lại với đầu ngón cái tạo thành hình tròn chữ O tròn trịa.",
        "fingers_state": {"thumb": "tip_touch", "index": "circle", "middle": "circle", "ring": "circle", "pinky": "circle"},
        "wrist_orientation": "up"
    },
    "P": {
        "label": "P",
        "type": "alphabet",
        "description": "Tương tự chữ K nhưng cổ tay chúi chúc xuống dưới hướng xuống đất.",
        "fingers_state": {"thumb": "between", "index": "straight_down", "middle": "angled_down", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "down"
    },
    "Q": {
        "label": "Q",
        "type": "alphabet",
        "description": "Tương tự chữ G nhưng cổ tay chúc thẳng xuống đất.",
        "fingers_state": {"thumb": "down", "index": "down", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "down"
    },
    "R": {
        "label": "R",
        "type": "alphabet",
        "description": "Ngón trỏ và ngón giữa duỗi thẳng bắt chéo nhau (ngón giữa vắt qua ngón trỏ).",
        "fingers_state": {"thumb": "folded", "index": "crossed", "middle": "crossed", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "S": {
        "label": "S",
        "type": "alphabet",
        "description": "Nắm chặt cả 4 ngón tay thành quả đấm, ngón cái gập vắt ngang qua mặt ngoài của các ngón tay.",
        "fingers_state": {"thumb": "across_front", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "T": {
        "label": "T",
        "type": "alphabet",
        "description": "Ngón cái kẹp chồi lên giữa ngón trỏ và ngón giữa của nắm đấm.",
        "fingers_state": {"thumb": "between_index_middle", "index": "curled_over", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "U": {
        "label": "U",
        "type": "alphabet",
        "description": "Ngón trỏ và ngón giữa duỗi thẳng song song khép sát nhau chỉ lên trên, ngón cái giữ ngón áp út và út.",
        "fingers_state": {"thumb": "folded", "index": "straight", "middle": "straight", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "V": {
        "label": "V",
        "type": "alphabet",
        "description": "Ngón trỏ và ngón giữa duỗi thẳng tách hình chữ V, các ngón còn lại nắm chặt.",
        "fingers_state": {"thumb": "folded", "index": "open_v", "middle": "open_v", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "W": {
        "label": "W",
        "type": "alphabet",
        "description": "Ba ngón trỏ, giữa và áp út duỗi thẳng xòe đều hình chữ W, ngón cái giữ ngón út.",
        "fingers_state": {"thumb": "folded", "index": "spread_w", "middle": "spread_w", "ring": "spread_w", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "X": {
        "label": "X",
        "type": "alphabet",
        "description": "Ngón trỏ uốn cong hình móc câu (gập đốt giữa và mút), các ngón còn lại nắm chặt.",
        "fingers_state": {"thumb": "folded", "index": "hooked", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    },
    "Y": {
        "label": "Y",
        "type": "alphabet",
        "description": "Ngón cái và ngón út duỗi sang hai bên, ba ngón giữa nắm vào lòng bàn tay (tư thế Shaka).",
        "fingers_state": {"thumb": "spread", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "spread"},
        "wrist_orientation": "up"
    },
    "Z": {
        "label": "Z",
        "type": "alphabet",
        "description": "Ngón trỏ duỗi thẳng vẽ nét chữ Z trong không gian.",
        "fingers_state": {"thumb": "folded", "index": "trace_z", "middle": "curled", "ring": "curled", "pinky": "curled"},
        "wrist_orientation": "up"
    }
}

VSL_VOCABULARY = {
    "XIN_CHAO": {
        "label": "Xin chào",
        "gloss": "XIN_CHAO",
        "type": "phrase",
        "keywords": ["xin chào", "chào", "hello", "hi", "chào bạn", "xin chao"],
        "description": "Thân người hơi nghiêng về trước, đầu gật chào; tay phải mở từ thái dương vẫy/hướng ra phía trước người đối diện, tay trái mở nâng ngang eo đón tiếp lịch sự.",
        "motion": "temple_to_forward_and_welcoming",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [20, 45, 10],
            "right_wrist": [15, 0, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "CAM_ON": {
        "label": "Cảm ơn",
        "gloss": "CAM_ON",
        "type": "phrase",
        "keywords": ["cảm ơn", "cám ơn", "thank you", "thanks", "đa tạ", "cam on"],
        "description": "Thân người cúi nhẹ 10 độ, đầu gật kính cẩn; các ngón tay phải khép phẳng chạm nhẹ cằm rồi đưa ra phía trước mở lòng bàn tay, tay trái ngửa mở phẳng đỡ phía dưới làm bệ đón biết ơn.",
        "motion": "chin_to_forward_with_bow",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [15, 30, 0],
            "right_wrist": [0, 10, -20],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "XIN_VUI_LONG": {
        "label": "Xin vui lòng",
        "gloss": "XIN_VUI_LONG",
        "type": "phrase",
        "keywords": ["xin vui lòng", "làm ơn", "vui lòng", "please", "nhờ", "giúp giùm", "xin lam on"],
        "description": "Thân người thẳng, đầu hơi nghiêng nhẹ cầu thị; lòng bàn tay phải mở phẳng áp lên ngực trái xoa tròn theo chiều kim đồng hồ, tay trái khép tự nhiên ngang eo.",
        "motion": "chest_circular_rub",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [25, -15, 0],
            "right_wrist": [0, -10, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "TAM_BIET": {
        "label": "Tạm biệt",
        "gloss": "TAM_BIET",
        "type": "phrase",
        "keywords": ["tạm biệt", "bye", "chào tạm biệt", "hẹn gặp lại", "tam biet"],
        "description": "Bàn tay nâng ngang vai, lòng bàn tay hướng về phía trước và gập các ngón tay lên xuống vẫy chào tạm biệt liên tục.",
        "motion": "wave_fingers",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [45, 45, 10],
            "right_wrist": [25, 0, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "XIN_LOI": {
        "label": "Xin lỗi",
        "gloss": "XIN_LOI",
        "type": "phrase",
        "keywords": ["xin lỗi", "lỗi", "sorry", "thứ lỗi", "xin loi"],
        "description": "Bàn tay nắm nhẹ thành nắm đấm áp vào vùng ngực trái và xoay tròn nhẹ nhàng thể hiện sự ăn năn thành khẩn, đầu hơi cúi.",
        "motion": "chest_circle",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [20, 20, -15],
            "right_wrist": [0, -10, 0],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "DONG_Y": {
        "label": "Đồng ý",
        "gloss": "DONG_Y",
        "type": "phrase",
        "keywords": ["đồng ý", "yes", "đúng", "chấp thuận", "được", "ok"],
        "description": "Bàn tay nắm lại thành nắm đấm giơ trước ngực, khẽ gật cổ tay lên xuống như đầu gật đồng ý.",
        "motion": "wrist_nod",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [30, 10, 0],
            "right_wrist": [30, 0, 0],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "KHONG": {
        "label": "Không",
        "gloss": "KHONG",
        "type": "phrase",
        "keywords": ["không", "no", "không đồng ý", "chưa", "không phải"],
        "description": "Ngón trỏ duỗi thẳng chỉ lên trên lắc qua lại sang hai bên, hoặc ngón trỏ và giữa chụm nhanh vào ngón cái biểu thị phủ định.",
        "motion": "finger_shake",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [35, 15, 0],
            "right_wrist": [0, 25, 0],
            "fingers": [0, 1, 0, 0, 0]
        }
    },
    "YEU_THUONG": {
        "label": "Yêu thương",
        "gloss": "YEU_THUONG",
        "type": "phrase",
        "keywords": ["yêu", "thương", "yêu thương", "love", "thích"],
        "description": "Hai cánh tay bắt chéo trước ngực, hai bàn tay áp vào hai vai hoặc ngực thể hiện sự ôm ấp yêu thương; hoặc cử chỉ 'I Love You' (ngón cái, trỏ và út duỗi).",
        "motion": "cross_chest_or_ily",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [35, 20, 0],
            "right_wrist": [10, 0, 0],
            "fingers": [1, 1, 0, 0, 1]
        }
    },
    "GIUP_DO": {
        "label": "Giúp đỡ",
        "gloss": "GIUP_DO",
        "type": "phrase",
        "keywords": ["giúp đỡ", "giúp", "cứu", "hỗ trợ", "help"],
        "description": "Lòng bàn tay trái ngửa mở phẳng đỡ dưới bàn tay phải đang nắm lại với ngón cái chỉ lên (tư thế Like), cả hai tay cùng nâng nhẹ lên phía trước.",
        "motion": "lift_upwards",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [25, 10, 0],
            "left_arm": [20, 10, 0],
            "fingers": [1, 0, 0, 0, 0]
        }
    },
    "TOI": {
        "label": "Tôi",
        "gloss": "TOI",
        "type": "phrase",
        "keywords": ["tôi", "mình", "em", "anh", "bản thân", "ta"],
        "description": "Ngón tay trỏ duỗi thẳng chỉ nhẹ vào giữa ngực của chính mình.",
        "motion": "point_to_self",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [25, -15, 0],
            "right_wrist": [-20, 0, 0],
            "fingers": [0, 1, 0, 0, 0]
        }
    },
    "BAN": {
        "label": "Bạn",
        "gloss": "BAN",
        "type": "phrase",
        "keywords": ["bạn", "người ấy", "cậu", "anh", "chị", "ông", "bà"],
        "description": "Ngón tay trỏ duỗi thẳng chỉ nhẹ về phía người đối diện.",
        "motion": "point_to_other",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [45, 15, 0],
            "right_wrist": [15, 0, 0],
            "fingers": [0, 1, 0, 0, 0]
        }
    },
    "AN_COM": {
        "label": "Ăn cơm",
        "gloss": "AN_COM",
        "type": "phrase",
        "keywords": ["ăn", "ăn cơm", "đói", "bữa ăn", "eat"],
        "description": "Các đầu ngón tay chụm lại đưa lên gần miệng 2-3 lần liên tiếp như động tác đưa thức ăn vào miệng.",
        "motion": "mouth_tap",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [15, 20, 0],
            "right_wrist": [0, -15, 0],
            "fingers": [0.5, 0.5, 0.5, 0.5, 0.5]
        }
    },
    "UONG_NUOC": {
        "label": "Uống nước",
        "gloss": "UONG_NUOC",
        "type": "phrase",
        "keywords": ["uống", "uống nước", "khát", "nước", "drink"],
        "description": "Bàn tay khum lại hình chữ C như đang cầm chiếc cốc, đưa lên miệng và ngửa cổ tay nhẹ nhàng như đang uống nước.",
        "motion": "cup_tilt",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [20, 25, 0],
            "right_wrist": [20, 0, 0],
            "fingers": [0.7, 0.7, 0.7, 0.7, 0.7]
        }
    },
    "BENH_VIEN": {
        "label": "Bệnh viện",
        "gloss": "BENH_VIEN",
        "type": "phrase",
        "keywords": ["bệnh viện", "bác sĩ", "y tá", "khám bệnh", "hospital"],
        "description": "Dùng ngón trỏ và ngón giữa bàn tay phải vẽ hình dấu chữ thập (chữ thập đỏ) trên bắp tay hoặc mu bàn tay trái.",
        "motion": "cross_trace",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [30, 0, -10],
            "left_arm": [25, 0, 10],
            "fingers": [0, 1, 1, 0, 0]
        }
    },
    "GIA_DINH": {
        "label": "Gia đình",
        "gloss": "GIA_DINH",
        "type": "phrase",
        "keywords": ["gia đình", "nhà", "family", "gia dinh"],
        "description": "Hai bàn tay tạo hình chữ F hoặc vòng khép, bắt đầu chạm nhẹ vào nhau trước ngực rồi vẽ một vòng tròn ra ngoài và gặp lại nhau thể hiện sự sum họp gia đình trọn vẹn.",
        "motion": "circle_and_meet",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [30, 15, 0],
            "left_arm": [30, -15, 0],
            "fingers": [1, 1, 0, 0, 0]
        }
    },
    "HOC_BAI": {
        "label": "Học bài",
        "gloss": "HOC_BAI",
        "type": "phrase",
        "keywords": ["học bài", "học", "học tập", "ôn bài", "làm bài", "study", "hoc bai", "hoc tap", "hoc"],
        "description": "Bàn tay trái mở phẳng ngửa như cuốn sách, tay phải làm động tác lấy kiến thức từ bàn tay trái vỗ nhẹ lên trán 2 lần thể hiện tiếp thu tri thức; đầu gật chăm chú.",
        "motion": "book_to_forehead",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [40, 20, 10],
            "left_arm": [25, 0, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "NAU_AN": {
        "label": "Nấu ăn",
        "gloss": "NAU_AN",
        "type": "phrase",
        "keywords": ["nấu ăn", "nấu", "nấu nướng", "làm cơm", "làm đồ ăn", "cook", "nau an", "nau"],
        "description": "Bàn tay trái khum nhẹ làm lòng chảo/nồi, bàn tay phải nắm hờ xoay tròn đảo qua lại như đang cầm muôi xào nấu thức ăn; thân người hơi nghiêng về trước.",
        "motion": "stir_pot_circle",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [30, 10, 0],
            "left_arm": [25, 0, 0],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "LAU_NHA": {
        "label": "Lau nhà",
        "gloss": "LAU_NHA",
        "type": "phrase",
        "keywords": ["lau nhà", "lau", "quét nhà", "lau dọn", "dọn dẹp", "dọn nhà", "mop", "clean", "lau nha", "quet nha"],
        "description": "Hai bàn tay nắm chặt trước ngực như đang cầm cán cây lau nhà, chuyển động đẩy tới lui nhịp nhàng từ trái qua phải; thân người hơi cúi theo nhịp lau sàn.",
        "motion": "mop_push_pull",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [35, 10, 0],
            "left_arm": [30, 0, 0],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "DI_NGU": {
        "label": "Đi ngủ",
        "gloss": "DI_NGU",
        "type": "phrase",
        "keywords": ["đi ngủ", "ngủ", "ngủ say", "buồn ngủ", "sleep", "di ngu", "ngu"],
        "description": "Hai lòng bàn tay khép phẳng áp vào nhau rồi nghiêng tựa vào một bên má, đầu nghiêng tựa vào tay mắt khẽ nhắm; biểu thị giấc ngủ yên bình.",
        "motion": "hands_on_cheek_tilt",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [35, 30, 15],
            "left_arm": [30, 20, 10],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "RUA_TAY": {
        "label": "Rửa tay",
        "gloss": "RUA_TAY",
        "type": "phrase",
        "keywords": ["rửa tay", "rửa", "vệ sinh tay", "wash hands", "rua tay", "rua"],
        "description": "Hai lòng bàn tay áp vào nhau xoa miết liên tục từ trước ra sau như đang rửa tay sạch sẽ dưới vòi nước.",
        "motion": "rub_hands_together",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [25, 0, 0],
            "left_arm": [25, 0, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "DOC_SACH": {
        "label": "Đọc sách",
        "gloss": "DOC_SACH",
        "type": "phrase",
        "keywords": ["đọc sách", "đọc", "xem sách", "read", "doc sach", "doc"],
        "description": "Hai bàn tay mở phẳng đặt cạnh nhau trước ngực như hai trang sách mở, đầu cúi nhẹ nhìn chăm chú vào trang sách.",
        "motion": "open_book_look",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [30, -10, 0],
            "left_arm": [30, 10, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "VIET_BAI": {
        "label": "Viết bài",
        "gloss": "VIET_BAI",
        "type": "phrase",
        "keywords": ["viết bài", "viết", "chép bài", "ghi chép", "write", "viet bai", "viet"],
        "description": "Bàn tay trái mở phẳng làm trang giấy, ngón cái và ngón trỏ tay phải chụm lại như cầm bút di chuyển viết lướt qua lại trên lòng bàn tay trái.",
        "motion": "pen_write_on_palm",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [25, 0, 0],
            "left_arm": [25, 0, 0],
            "fingers": [0, 1, 0, 0, 0]
        }
    },
    "LAM_VIEC": {
        "label": "Làm việc",
        "gloss": "LAM_VIEC",
        "type": "phrase",
        "keywords": ["làm việc", "đi làm", "làm", "công việc", "work", "lam viec", "di lam", "lam"],
        "description": "Hai bàn tay nắm lại gõ lên xuống luân phiên thể hiện động tác làm việc chăm chỉ, tích cực.",
        "motion": "fists_alternate_tap",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [30, 10, 0],
            "left_arm": [30, -10, 0],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "DI_HOC": {
        "label": "Đi học",
        "gloss": "DI_HOC",
        "type": "phrase",
        "keywords": ["đi học", "tới trường", "đến lớp", "go to school", "di hoc"],
        "description": "Hai bàn tay ngửa mở phẳng như trang sách rồi đưa ra phía trước biểu thị đến trường học tập.",
        "motion": "book_forward",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [35, 10, 0],
            "left_arm": [35, -10, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "TAM": {
        "label": "Tắm",
        "gloss": "TAM",
        "type": "phrase",
        "keywords": ["tắm", "tắm rửa", "gội đầu", "shower", "tam", "tam rua"],
        "description": "Hai bàn tay nắm nhẹ xoa vuốt lên xuống hai bên ngực và vai như đang cọ rửa cơ thể khi tắm.",
        "motion": "chest_rub_wash",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [25, -10, 0],
            "left_arm": [25, 10, 0],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "KHOE_MANH": {
        "label": "Khỏe mạnh",
        "gloss": "KHOE_MANH",
        "type": "phrase",
        "keywords": ["khỏe mạnh", "khỏe", "sức khỏe", "mạnh mẽ", "strong", "khoe manh", "khoe"],
        "description": "Hai cánh tay gập vuông góc 90 độ gồng cơ bắp trước ngực, thân người ưỡn thẳng tự tin mỉm cười.",
        "motion": "flex_biceps",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [45, 30, 10],
            "left_arm": [45, -30, -10],
            "fingers": [0, 0, 0, 0, 0]
        }
    },
    "MET_MOI": {
        "label": "Mệt mỏi",
        "gloss": "MET_MOI",
        "type": "phrase",
        "keywords": ["mệt mỏi", "mệt", "đuối", "kiệt sức", "tired", "met moi", "met"],
        "description": "Hai cánh tay buông thõng, thân người khom rũ xuống phía trước, đầu cúi thở dài thể hiện sự kiệt sức.",
        "motion": "body_slump_exhausted",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [10, 0, 0],
            "left_arm": [10, 0, 0],
            "fingers": [0.3, 0.3, 0.3, 0.3, 0.3]
        }
    },
    "VUI_VE": {
        "label": "Vui vẻ",
        "gloss": "VUI_VE",
        "type": "phrase",
        "keywords": ["vui vẻ", "vui", "mừng", "hạnh phúc", "happy", "vui ve", "vui"],
        "description": "Hai bàn tay mở rộng trước ngực vẫy rung nhẹ hướng lên trên, đầu ngẩng cao mỉm cười rạng rỡ biểu thị niềm vui sướng.",
        "motion": "chest_flutter_up",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [35, 15, 0],
            "left_arm": [35, -15, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "BUON": {
        "label": "Buồn",
        "gloss": "BUON",
        "type": "phrase",
        "keywords": ["buồn", "đau buồn", "buồn bã", "thất vọng", "sad", "buon"],
        "description": "Bàn tay phải mở phẳng vuốt từ ngực dọc xuống bụng, đầu cúi ủ rũ thể hiện nỗi buồn phiền trong lòng.",
        "motion": "hand_slide_down_chest",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [20, -10, 0],
            "right_wrist": [-15, 0, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "HOI": {
        "label": "Hỏi",
        "gloss": "HOI",
        "type": "phrase",
        "keywords": ["hỏi", "thắc mắc", "câu hỏi", "ask", "question", "hoi"],
        "description": "Ngón trỏ tay phải vẽ hình dấu chấm hỏi trong không gian rồi dừng lại chấm một điểm hướng về phía người đối diện.",
        "motion": "trace_question_mark",
        "two_handed": False,
        "avatar_pose": {
            "right_arm": [35, 15, 0],
            "right_wrist": [15, 0, 0],
            "fingers": [0, 1, 0, 0, 0]
        }
    },
    "TRA_LOI": {
        "label": "Trả lời",
        "gloss": "TRA_LOI",
        "type": "phrase",
        "keywords": ["trả lời", "giải đáp", "đáp lại", "answer", "tra loi", "dap"],
        "description": "Hai ngón trỏ bắt đầu từ gần cằm/miệng rồi cùng chỉ hướng thẳng ra phía trước về phía người đối diện biểu thị câu trả lời.",
        "motion": "chin_point_forward",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [30, 5, 0],
            "left_arm": [30, -5, 0],
            "fingers": [0, 1, 0, 0, 0]
        }
    },
    "CHO": {
        "label": "Cho",
        "gloss": "CHO",
        "type": "phrase",
        "keywords": ["cho", "tặng", "biếu", "gửi", "give", "cho", "tang"],
        "description": "Hai bàn tay ngửa mở phẳng từ ngực đưa thẳng ra phía trước thể hiện lòng hảo tâm trao tặng.",
        "motion": "both_hands_extend_forward",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [35, 10, 0],
            "left_arm": [35, -10, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "XIN": {
        "label": "Xin",
        "gloss": "XIN",
        "type": "phrase",
        "keywords": ["xin", "nhận", "lấy", "get", "receive", "xin", "nhan"],
        "description": "Hai bàn tay ngửa mở từ phía trước kéo thu nhẹ về phía ngực biểu thị tiếp nhận điều gì đó.",
        "motion": "both_hands_pull_in",
        "two_handed": True,
        "avatar_pose": {
            "right_arm": [20, 5, 0],
            "left_arm": [20, -5, 0],
            "fingers": [1, 1, 1, 1, 1]
        }
    },
    "CUOI_TUAN": {
        "label": "Cuối tuần",
        "gloss": "CUOI_TUAN",
        "type": "phrase",
        "keywords": ["cuối tuần", "ngày cuối tuần", "những ngày cuối tuần", "weekend", "cuoi tuan"],
        "description": "Hai bàn tay dựng phẳng trước ngực, tay phải quét ngang từ trái sang phải rồi gập chéo biểu thị kết thúc một tuần.",
        "motion": "weekend_sweep_and_finish",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 20, 0], "left_arm": [25, -20, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "LICH_TRINH": {
        "label": "Lịch trình",
        "gloss": "LICH_TRINH",
        "type": "phrase",
        "keywords": ["lịch trình", "lịch trình cụ thể", "kế hoạch", "thời gian biểu", "schedule", "lich trinh", "ke hoach"],
        "description": "Tay trái xòe phẳng làm mặt bảng biểu; tay phải dùng ngón trỏ và ngón giữa vạch các hàng cột có trật tự trên lòng bàn tay trái.",
        "motion": "grid_schedule_trace",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 10, 0], "left_arm": [30, -15, 0], "fingers": [0, 1, 1, 0, 0] }
    },
    "PHU_THUOC": {
        "label": "Phụ thuộc",
        "gloss": "PHU_THUOC",
        "type": "phrase",
        "keywords": ["phụ thuộc", "tùy thuộc", "tùy theo", "phụ thuộc vào", "depend", "phu thuoc", "tuy thuoc"],
        "description": "Hai ngón trỏ móc vào nhau đong đưa nhẹ thể hiện sự gắn kết và phụ thuộc lẫn nhau giữa 2 sự việc.",
        "motion": "hooked_index_swing",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 5, 0], "left_arm": [30, -5, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "BAN_RON": {
        "label": "Bận rộn",
        "gloss": "BAN_RON",
        "type": "phrase",
        "keywords": ["bận rộn", "mức độ bận rộn", "bận", "busy", "ban ron", "ban"],
        "description": "Hai bàn tay mở phẳng đảo qua đảo lại thật nhanh trước ngực thể hiện công việc dồn dập tấp nập.",
        "motion": "rapid_cross_hands_busy",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 15, 0], "left_arm": [35, -15, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "TRONG_TUAN": {
        "label": "Trong tuần",
        "gloss": "TRONG_TUAN",
        "type": "phrase",
        "keywords": ["trong tuần", "các ngày trong tuần", "ngày trong tuần", "weekday", "trong tuan"],
        "description": "Tay trái ngửa phẳng; ngón trỏ tay phải trượt thẳng từ cổ tay trái đến các đầu ngón tay trái biểu thị dòng chảy của một tuần.",
        "motion": "week_slide_forward",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 10, 0], "left_arm": [25, -20, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "THOI_GIAN": {
        "label": "Thời gian",
        "gloss": "THOI_GIAN",
        "type": "phrase",
        "keywords": ["thời gian", "dành thời gian", "giờ giấc", "time", "thoi gian"],
        "description": "Ngón trỏ tay phải gõ nhịp 2 lần lên cổ tay trái nơi đeo đồng hồ thể hiện khái niệm thời gian.",
        "motion": "tap_wrist_watch",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, -10, 0], "left_arm": [25, -25, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "IT": {
        "label": "Một ít",
        "gloss": "IT",
        "type": "phrase",
        "keywords": ["một ít", "chút ít", "ít", "chút", "little", "few", "mot it", "it"],
        "description": "Đầu ngón cái và đầu ngón trỏ tay phải khép sát nhau chừa một khe hẹp nhỏ xíu, mắt nheo mỉm cười biểu thị số lượng ít ỏi.",
        "motion": "pinch_small_gap",
        "two_handed": False,
        "avatar_pose": { "right_arm": [35, 15, 0], "left_arm": [20, -10, 0], "fingers": [1, 1, 0, 0, 0] }
    },
    "DU_AN": {
        "label": "Dự án",
        "gloss": "DU_AN",
        "type": "phrase",
        "keywords": ["dự án", "dự án cá nhân", "project", "kế hoạch lớn", "du an"],
        "description": "Hai bàn tay tạo chữ P/D xoay vòng tròn định hình một đề án công việc trước ngực.",
        "motion": "project_outline_box",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 15, 0], "left_arm": [35, -15, 0], "fingers": [1, 1, 0, 0, 0] }
    },
    "CA_NHAN": {
        "label": "Cá nhân",
        "gloss": "CA_NHAN",
        "type": "phrase",
        "keywords": ["cá nhân", "riêng mình", "bản thân", "personal", "self", "ca nhan"],
        "description": "Bàn tay phải mở khum nhẹ vỗ nhẹ vào ngực của chính mình 2 lần biểu thị tính cá nhân riêng tư.",
        "motion": "tap_chest_self",
        "two_handed": False,
        "avatar_pose": { "right_arm": [25, 0, 0], "left_arm": [20, -10, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "DAM_BAO": {
        "label": "Đảm bảo",
        "gloss": "DAM_BAO",
        "type": "phrase",
        "keywords": ["đảm bảo", "chắc chắn", "cam kết", "guarantee", "sure", "dam bao"],
        "description": "Nắm tay phải gõ dứt khoát xuống lòng bàn tay trái mở ngửa, đầu gật dứt khoát khẳng định chắc chắn.",
        "motion": "fist_strike_open_palm",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 0, 0], "left_arm": [25, -20, 0], "fingers": [0, 0, 0, 0, 0] }
    },
    "DU": {
        "label": "Đủ",
        "gloss": "DU",
        "type": "phrase",
        "keywords": ["đủ", "có đủ", "đầy đủ", "enough", "du", "co du"],
        "description": "Nắm tay trái dựng đứng; lòng bàn tay phải vuốt phẳng qua đỉnh nắm tay trái như gạt một chén nước đầy vừa đủ.",
        "motion": "sweep_over_fist_enough",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 5, 0], "left_arm": [25, -15, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "THU_GIAN": {
        "label": "Thư giãn",
        "gloss": "THU_GIAN",
        "type": "phrase",
        "keywords": ["thư giãn", "nghỉ ngơi", "xả hơi", "relax", "rest", "thu gian", "nghi ngoi"],
        "description": "Hai bàn tay thả lỏng buông xuống nhẹ nhàng, vai hạ thấp, đầu ngửa nhẹ thư thái hít thở sâu.",
        "motion": "hands_glide_down_relax",
        "two_handed": True,
        "avatar_pose": { "right_arm": [15, 10, 0], "left_arm": [15, -10, 0], "fingers": [0.5, 0.5, 0.5, 0.5, 0.5] }
    },
    "NEU": {
        "label": "Nếu",
        "gloss": "NEU",
        "type": "phrase",
        "keywords": ["nếu", "nếu có thể", "giả sử", "if", "neu", "neu co the"],
        "description": "Ngón út tay phải chấm nhẹ dưới khóe mắt hoặc thái dương 2 lần với vẻ mặt suy ngẫm giả định.",
        "motion": "pinky_tap_cheek_if",
        "two_handed": False,
        "avatar_pose": { "right_arm": [30, 20, 0], "left_arm": [20, -10, 0], "fingers": [0, 0, 0, 0, 1] }
    },
    "CO_THE": {
        "label": "Có thể",
        "gloss": "CO_THE",
        "type": "phrase",
        "keywords": ["có thể", "được", "khả năng", "can", "possible", "co the"],
        "description": "Hai nắm tay hướng về trước rồi cùng gật nhẹ xuống 2 nhịp dứt khoát thể hiện sự có thể làm được.",
        "motion": "both_fists_nod_down",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 15, 0], "left_arm": [30, -15, 0], "fingers": [0, 0, 0, 0, 0] }
    },
    "THICH": {
        "label": "Thích",
        "gloss": "THICH",
        "type": "phrase",
        "keywords": ["thích", "yêu thích", "khoái", "like", "thich", "yeu thich"],
        "description": "Ngón cái và ngón giữa tay phải chạm vào giữa ngực rồi khép lại kéo nhẹ ra ngoài với nụ cười sảng khoái.",
        "motion": "thumb_middle_pull_chest_like",
        "two_handed": False,
        "avatar_pose": { "right_arm": [30, 0, 0], "left_arm": [20, -10, 0], "fingers": [1, 0, 1, 0, 0] }
    },
    "GAP_GO": {
        "label": "Gặp gỡ",
        "gloss": "GAP_GO",
        "type": "phrase",
        "keywords": ["gặp gỡ", "gặp", "hội ngộ", "meet", "gap go", "gap"],
        "description": "Hai ngón trỏ từ hai bên đưa vào chính giữa gặp nhau đối đầu biểu thị hai người gặp gỡ.",
        "motion": "two_index_fingers_meet",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 10, 0], "left_arm": [35, -10, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "BAN_BE": {
        "label": "Bạn bè",
        "gloss": "BAN_BE",
        "type": "phrase",
        "keywords": ["bạn bè", "bạn", "bạn thân", "friend", "ban be", "ban than"],
        "description": "Hai ngón trỏ móc vào nhau rồi đảo chiều móc lại lần nữa thể hiện tình bạn khăng khít gắn bó.",
        "motion": "interlock_index_friends",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 5, 0], "left_arm": [30, -5, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "KHAM_PHA": {
        "label": "Khám phá",
        "gloss": "KHAM_PHA",
        "type": "phrase",
        "keywords": ["khám phá", "tìm hiểu", "trải nghiệm", "đi xem", "explore", "discover", "kham pha"],
        "description": "Bàn tay phải khum hình chữ C trước mắt rồi đưa vòng ra trước và mở bung các ngón tay thể hiện ánh mắt mở rộng nhìn chân trời mới.",
        "motion": "explore_vision_expand",
        "two_handed": False,
        "avatar_pose": { "right_arm": [35, 15, 0], "left_arm": [20, -10, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "DIA_DIEM": {
        "label": "Địa điểm",
        "gloss": "DIA_DIEM",
        "type": "phrase",
        "keywords": ["địa điểm", "địa điểm mới", "nơi", "nơi chốn", "chỗ", "place", "location", "dia diem"],
        "description": "Hai bàn tay tạo hình tròn úp song song rồi ngón trỏ phải chỉ cắm xuống tâm biểu thị vị trí địa điểm xác định.",
        "motion": "circle_and_point_place",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 10, 0], "left_arm": [30, -15, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "MOI": {
        "label": "Mới",
        "gloss": "MOI",
        "type": "phrase",
        "keywords": ["mới", "mới lạ", "tươi mới", "new", "moi", "moi la"],
        "description": "Mu bàn tay phải ngửa vuốt từ mu bàn tay trái hất nhẹ lên cao tươi sáng biểu thị sự mới mẻ mọc lên.",
        "motion": "back_hand_sweep_up_new",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 10, 0], "left_arm": [25, -20, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "CO_HOI": {
        "label": "Cơ hội",
        "gloss": "CO_HOI",
        "type": "phrase",
        "keywords": ["cơ hội", "dịp", "thời cơ", "opportunity", "chance", "co hoi"],
        "description": "Hai bàn tay mở ngửa từ dưới chộp nắm lại đưa lên ngực thể hiện nắm bắt lấy cơ hội tốt.",
        "motion": "catch_opportunity_hands",
        "two_handed": True,
        "avatar_pose": { "right_arm": [35, 10, 0], "left_arm": [35, -10, 0], "fingers": [1, 1, 1, 1, 1] }
    },
    "THOI_QUEN": {
        "label": "Thói quen",
        "gloss": "THOI_QUEN",
        "type": "phrase",
        "keywords": ["thói quen", "thói quen thường ngày", "lề lối", "habit", "thoi quen"],
        "description": "Cổ tay phải gập đè lên cổ tay trái rồi cả hai cùng hạ xuống 2 lần biểu thị nếp quen cố định.",
        "motion": "wrists_cross_press_habit",
        "two_handed": True,
        "avatar_pose": { "right_arm": [30, 5, 0], "left_arm": [25, -10, 0], "fingers": [0, 0, 0, 0, 0] }
    },
    "HANG_NGAY": {
        "label": "Hàng ngày",
        "gloss": "HANG_NGAY",
        "type": "phrase",
        "keywords": ["hàng ngày", "thường ngày", "mỗi ngày", "daily", "everyday", "hang ngay", "thuong ngay"],
        "description": "Mu bàn tay phải quẹt nhẹ qua má từ sau ra trước 2 lần thể hiện ngày này qua ngày khác diễn ra đều đặn.",
        "motion": "knuckle_stroke_cheek_daily",
        "two_handed": False,
        "avatar_pose": { "right_arm": [30, 20, 0], "left_arm": [20, -10, 0], "fingers": [0, 0, 0, 0, 0] }
    },
    "TOI": {
        "label": "Tôi",
        "gloss": "TOI",
        "type": "phrase",
        "keywords": ["tôi", "của tôi", "mình", "em", "anh", "i", "me", "my", "toi"],
        "description": "Ngón trỏ tay phải chỉ nhẹ vào giữa ngực của mình thể hiện ngôi thứ nhất bản thân.",
        "motion": "point_to_self_chest",
        "two_handed": False,
        "avatar_pose": { "right_arm": [25, 0, 0], "left_arm": [20, -10, 0], "fingers": [0, 1, 0, 0, 0] }
    },
    "KHONG": {
        "label": "Không",
        "gloss": "KHONG",
        "type": "phrase",
        "keywords": ["không", "không có", "chưa", "chẳng", "no", "not", "khong", "khong co"],
        "description": "Bàn tay phải mở lắc qua lại sang 2 bên kết hợp đầu lắc nhẹ dứt khoát biểu thị sự phủ định.",
        "motion": "hand_and_head_shake_no",
        "two_handed": False,
        "avatar_pose": { "right_arm": [35, 15, 0], "left_arm": [20, -10, 0], "fingers": [1, 1, 1, 1, 1] }
    }
}

ALL_SIGNS = {**VSL_ALPHABET, **VSL_VOCABULARY}
