/**
 * Quy ước ký hiệu và Provenance chuẩn hóa cho 7 chữ cái có dấu VSL.
 * =================================================================
 * Thứ tự ưu tiên bắt buộc: TOÀN QUỐC -> MIỀN NAM.
 * Tuyệt đối không sử dụng video hậu tố B (Miền Bắc) hoặc T (Miền Trung) làm baseline chính.
 * Nếu thiếu video Toàn quốc hoặc Miền Nam: gắn cờ BLOCKED_SOURCE.
 */

export const SIGN_CONVENTIONS = Object.freeze({
  'Đ': {
    letter: 'Đ',
    baseLetter: 'D',
    status: 'VERIFIED_NATIONAL',
    statusBadge: 'TOÀN QUỐC',
    statusLabel: 'Toàn quốc (QIPEDC Chuẩn D0495)',
    videoId: 'D0495',
    videoFilename: 'D0495.mp4',
    videoUrl: '/api/media/videos/D0495.mp4',
    provenancePolicy: 'TOÀN QUỐC',
    description: 'Ngón trỏ giơ lên (hơi khum móc nhẹ), ngón cái chạm cụm ngón giữa-nhẫn-út (chữ D). Sau đó ngón trỏ gập gãy nhanh tại đốt giữa (PIP joint) xuống sâu hơn tạo nét gạch ngang của chữ Đ.',
    steps: {
      startingPose: 'Ngón trỏ giơ lên (thẳng hoặc hơi khum nhẹ ~125°), ngón cái chạm đầu ngón giữa, ngón áp út và út gập ôm theo (chữ nền D chuẩn).',
      fingersHeld: 'Ngón cái giữ tiếp xúc cố định với cụm 3 ngón (giữa, nhẫn, út); cổ tay giữ ổn định trong khung hình.',
      movement: 'Ngón trỏ gập nhanh và dứt khoát tại đốt giữa (PIP joint) xuống sâu hơn tạo nét móc ngang của chữ Đ.',
      endingPose: 'Ngón trỏ ở trạng thái gập móc cong sâu (PIP < 115°), các ngón còn lại giữ nguyên tư thế D.',
      antiConfusion: 'Không vẩy cả bàn tay hoặc di chuyển cổ tay để tránh nhận nhầm thành rung lắc; gập riêng biệt đốt ngón trỏ.'
    }
  },
  'Ô': {
    letter: 'Ô',
    baseLetter: 'O',
    status: 'VERIFIED_SOUTHERN',
    statusBadge: 'MIỀN NAM',
    statusLabel: 'Miền Nam (QIPEDC D0507N)',
    videoId: 'D0507N',
    videoFilename: 'D0507N.mp4',
    videoUrl: '/api/media/videos/D0507N.mp4',
    provenancePolicy: 'TOÀN QUỐC -> MIỀN NAM',
    description: 'Bàn tay khum tròn tạo hình chữ O. Sau đó ngón trỏ và ngón giữa cùng mở tách nhẹ nhau ra, hơi khum cong hướng vào tạo hình dấu mũ ∧ (circumflex claw).',
    steps: {
      startingPose: 'Bàn tay khum tròn tạo hình chữ O (đầu ngón cái chạm đầu các ngón tay tạo khuyên kín).',
      fingersHeld: 'Ngón áp út và ngón út giữ gập cong ôm theo lòng bàn tay bên dưới.',
      movement: 'Ngón trỏ và ngón giữa cùng bung mở tách nhẹ nhau ra, hơi khum cong tại đốt giữa tạo hình dấu mũ ∧.',
      endingPose: 'Ngón trỏ và ngón giữa xòe hình dấu mũ hơi cong hướng lên/ra trước, ngón cái-áp út-út giữ khum bên dưới.',
      antiConfusion: 'Tuyệt đối KHÔNG nâng cả bàn tay lên cao (quy ước Miền Bắc D0507B đã bị loại bỏ); không bật duỗi ngón út (tránh nhầm chữ Ơ).'
    }
  },
  'Ơ': {
    letter: 'Ơ',
    baseLetter: 'O',
    status: 'VERIFIED_SOUTHERN',
    statusBadge: 'MIỀN NAM',
    statusLabel: 'Miền Nam (QIPEDC D0508N)',
    videoId: 'D0508N',
    videoFilename: 'D0508N.mp4',
    videoUrl: '/api/media/videos/D0508N.mp4',
    provenancePolicy: 'TOÀN QUỐC -> MIỀN NAM',
    description: 'Bàn tay khum tròn tạo hình chữ O. Sau đó cụm ngón trỏ-giữa-nhẫn khép gập chạm ngón cái, đồng thời bật ngón út vươn lên trên và hơi khum móc nhẹ tạo dấu móc râu của chữ Ơ (không lắc cổ tay).',
    steps: {
      startingPose: 'Bàn tay khum tròn tạo hình chữ O (ngón cái chạm các đầu ngón tay tạo vòng tròn).',
      fingersHeld: 'Ngón trỏ, ngón giữa và ngón áp út thu khép gập chạm gọn vào ngón cái; cổ tay giữ đứng yên.',
      movement: 'Bật dứt khoát ngón út vươn lên trên và hơi khum móc nhẹ đầu ngón út tạo hình dấu móc râu Ơ (KHÔNG lắc cổ tay).',
      endingPose: 'Ngón út vươn tách rõ rệt lên trên với đầu ngón hơi khum móc râu (d_PW ≥ 1.35), 4 ngón còn lại khép kín vào nhau.',
      antiConfusion: 'Không mở bung ngón trỏ và giữa (tránh nhầm chữ Ô); không duỗi xoè ngón cái ngang (tránh nhầm chữ Y); không lắc cổ tay.'
    }
  },
  'Ư': {
    letter: 'Ư',
    baseLetter: 'U',
    status: 'VERIFIED_SOUTHERN',
    statusBadge: 'MIỀN NAM',
    statusLabel: 'Miền Nam (QIPEDC D0515N)',
    videoId: 'D0515N',
    videoFilename: 'D0515N.mp4',
    videoUrl: '/api/media/videos/D0515N.mp4',
    provenancePolicy: 'TOÀN QUỐC -> MIỀN NAM',
    description: 'Ngón trỏ và ngón giữa giơ thẳng đứng song song (chữ U). Sau đó gập đồng thời cả ngón trỏ và ngón giữa xuống khép chạm vào ngón cái, đồng thời bật ngón út vươn lên và hơi khum móc nhẹ tạo dấu râu chữ Ư.',
    steps: {
      startingPose: 'Ngón trỏ và ngón giữa giơ thẳng đứng song song sát nhau lên trời, ngón áp út và út gập chạm ngón cái (chữ U chuẩn).',
      fingersHeld: 'Ngón áp út và ngón cái khép giữ bên dưới làm trụ; cổ tay giữ đứng yên ổn định.',
      movement: 'Gập cụp cả ngón trỏ và ngón giữa xuống chạm vào ngón cái, đồng thời bật ngón út vươn lên trên hơi khum móc râu tạo hình chữ Ư.',
      endingPose: 'Ngón trỏ và giữa đã gập xuống chụm với ngón cái, riêng ngón út bật vươn lên hơi khum móc râu (pinky_curl ≤ 0.35, d_PW ≥ 1.35).',
      antiConfusion: 'KHÔNG giữ thẳng ngón trỏ và giữa suốt động tác (khớp đúng video D0515N); không bung ngón cái duỗi ngang sang bên (tránh nhầm thành chữ Y).'
    }
  },
  'Ă': {
    letter: 'Ă',
    baseLetter: 'A',
    status: 'VERIFIED_SOUTHERN',
    statusBadge: 'MIỀN NAM',
    statusLabel: 'Miền Nam (QIPEDC D0490N)',
    videoId: 'D0490N',
    videoFilename: 'D0490N.mp4',
    videoUrl: '/api/media/videos/D0490N.mp4',
    provenancePolicy: 'TOÀN QUỐC -> MIỀN NAM',
    description: 'Nắm tay thành quả đấm (chữ A). Sau đó CHỈ bật duỗi thẳng 1 ngón trỏ đứng lên trời; ngón giữa, áp út và út vẫn giữ nắm gập chặt tạo nét trăng khuyết của chữ Ă.',
    steps: {
      startingPose: 'Nắm tay thành quả đấm (chữ A chuẩn), ngón cái áp sát cạnh ngón trỏ.',
      fingersHeld: 'Ngón giữa, ngón áp út và ngón út (cùng ngón cái) giữ gập nắm chặt cố định trong lòng bàn tay.',
      movement: 'CHỈ bung duỗi thẳng 1 ngón trỏ duy nhất hướng thẳng lên trên (d_IM tách xa khỏi ngón giữa đang nắm).',
      endingPose: 'Riêng ngón trỏ dựng thẳng đứng lên trên, ngón giữa và các ngón còn lại vẫn giữ nắm chặt hình quả đấm.',
      antiConfusion: 'KHÔNG bật cả ngón giữa lên cùng (bật trỏ+giữa sẽ là chữ Â hoặc V); tuyệt đối KHÔNG vẽ vòng cung trên không (quy ước Miền Bắc D0490B đã loại bỏ).'
    }
  },
  'Â': {
    letter: 'Â',
    baseLetter: 'A',
    status: 'VERIFIED_SOUTHERN',
    statusBadge: 'MIỀN NAM',
    statusLabel: 'Miền Nam (QIPEDC D0491N)',
    videoId: 'D0491N',
    videoFilename: 'D0491N.mp4',
    videoUrl: '/api/media/videos/D0491N.mp4',
    provenancePolicy: 'TOÀN QUỐC -> MIỀN NAM',
    description: 'Nắm tay thành quả đấm (chữ A). Sau đó bật mở ĐỒNG THỜI CẢ HAI ngón trỏ và ngón giữa tách nhẹ nhau ra, hơi khum cong nhẹ ở đốt giữa tạo hình dấu mũ ∧ (circumflex claw).',
    steps: {
      startingPose: 'Nắm tay thành quả đấm (chữ A chuẩn), ngón cái áp sát cạnh ngón trỏ.',
      fingersHeld: 'Ngón áp út và ngón út giữ gập chặt trong lòng bàn tay; ngón cái giữ cố định.',
      movement: 'Bật mở ĐỒNG THỜI CẢ ngón trỏ và ngón giữa hướng lên hơi tách nhẹ, hai ngón hơi khum cong nhẹ tạo hình dấu mũ ∧.',
      endingPose: 'Cả ngón trỏ và ngón giữa cùng mở tách nhẹ hơi cong tạo hình dấu mũ ∧, ngón áp út và út vẫn gập kín.',
      antiConfusion: 'Phân biệt với chữ Ă (chỉ giơ thẳng 1 ngón trỏ, ngón giữa vẫn nắm gập); tuyệt đối KHÔNG nâng cả nắm đấm lên cao (quy ước Miền Bắc D0491B đã loại bỏ).'
    }
  },
  'Ê': {
    letter: 'Ê',
    baseLetter: 'E',
    status: 'VERIFIED_SOUTHERN',
    statusBadge: 'MIỀN NAM',
    statusLabel: 'Miền Nam (QIPEDC D0497N)',
    videoId: 'D0497N',
    videoFilename: 'D0497N.mp4',
    videoUrl: '/api/media/videos/D0497N.mp4',
    provenancePolicy: 'TOÀN QUỐC -> MIỀN NAM',
    description: 'Bàn tay tạo dáng chữ E (các đầu ngón tay gập cong đè lên ngón cái). Sau đó ngón trỏ và ngón giữa cùng bung mở tách nhẹ nhau ra, hơi khum cong tạo hình dấu mũ ∧.',
    steps: {
      startingPose: 'Bàn tay tạo hình chữ E chuẩn: 4 ngón cong gập áp lên đầu ngón cái.',
      fingersHeld: 'Ngón áp út và út giữ cong ôm sát lòng bàn tay cạnh ngón cái.',
      movement: 'Ngón trỏ và ngón giữa cùng mở tách nhẹ nhau ra, hơi khum cong ở đốt giữa tạo hình dấu mũ ∧.',
      endingPose: 'Ngón trỏ và giữa xòe hình dấu mũ ∧ hơi cong, hai ngón áp út và út giữ nguyên tư thế gập cong chữ E.',
      antiConfusion: 'Tuyệt đối KHÔNG nâng cả bàn tay E lên cao (quy ước Miền Bắc D0497B đã bị loại bỏ); bung dứt khoát hai ngón trỏ và giữa tạo dấu mũ.'
    }
  }
});

