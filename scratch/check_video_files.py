import os

video_dir = "backend/data/media_cache/videos"
all_files = os.listdir(video_dir) if os.path.exists(video_dir) else []

for code in ["D0509", "D0510", "D0511", "D0517"]:
    matches = [f for f in all_files if f.startswith(code)]
    print(f"Code {code}: {matches}")
