import cv2

for idx in [32, 40, 48]:
    f = f'scratch/frames_d0539/frame_{idx:03d}.jpg'
    img = cv2.imread(f)
    # Vùng bàn tay người ký
    h, w, _ = img.shape
    # Toạ độ bàn tay xuất hiện quanh (w*0.4 to w*0.6, h*0.35 to h*0.65)
    hand_crop = img[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.6)]
    cv2.imwrite(f'scratch/frames_d0539/hand_close_{idx}.jpg', hand_crop)
    print(f"Saved hand_close_{idx}.jpg shape: {hand_crop.shape}")

