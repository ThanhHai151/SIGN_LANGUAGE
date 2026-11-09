import cv2

# Check image sizes and inspect
for name in ['hand_R_f28.jpg', 'hand_R_f34.jpg', 'hand_X_f28.jpg', 'hand_X_f36.jpg', 'hand_X_f112.jpg']:
    p = f'scratch/{name}'
    img = cv2.imread(p)
    if img is not None:
        print(f"{name}: shape = {img.shape}")
    else:
        print(f"Could not load {name}")
