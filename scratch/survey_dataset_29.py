"""
Script khảo sát toàn diện Backend, Frontend, Dataset cho bảng chữ cái tiếng Việt 29 chữ:
A, Ă, Â, B, C, D, Đ, E, Ê, G, H, I, K, L, M, N, O, Ô, Ơ, P, Q, R, S, T, U, Ư, V, X, Y.
"""
import os
import json
import numpy as np

WORKSPACE = "d:/SIGN_LANGUAGE"

# 1. Kiểm tra Dataset vsl_alphabet_static_v2.npz
npz_static_path = os.path.join(WORKSPACE, "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz")
if os.path.exists(npz_static_path):
    data_static = np.load(npz_static_path, allow_pickle=True)
    classes_static = list(data_static["classes"])
    y_static = data_static["y"]
    unique, counts = np.unique(y_static, return_counts=True)
    static_counts = {classes_static[u]: int(c) for u, c in zip(unique, counts)}
else:
    static_counts = {}
    classes_static = []

# 2. Kiểm tra Dataset vsl_dataset.npz nếu có
npz_vsl_path = os.path.join(WORKSPACE, "backend/data/vsl_dataset.npz")
vsl_dataset_info = {}
if os.path.exists(npz_vsl_path):
    d_vsl = np.load(npz_vsl_path, allow_pickle=True)
    vsl_dataset_info["keys"] = list(d_vsl.keys())
    if "classes" in d_vsl:
        vsl_dataset_info["classes"] = list(d_vsl["classes"])
    if "labels" in d_vsl:
        vsl_dataset_info["labels_len"] = len(d_vsl["labels"])
    if "y" in d_vsl:
        vsl_dataset_info["y_len"] = len(d_vsl["y"])

# 3. Kiểm tra vsl_alphabet_motion_v2.json (cho các chữ động Ă, Â, Ê, Ô, Ơ, Ư, Đ)
motion_json_path = os.path.join(WORKSPACE, "backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json")
motion_info = {}
if os.path.exists(motion_json_path):
    with open(motion_json_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    motion_info = {
        k: {
            "target": v.get("target_letter"),
            "base": v.get("base_handshape"),
            "video_id": v.get("video_id"),
            "total_frames": v.get("total_frames_in_video"),
            "valid_frames": v.get("total_valid_frames"),
            "diacritic_frames": len(v.get("diacritic_motion_frames", [])),
            "base_pose_frames": len(v.get("base_pose_frames", []))
        }
        for k, v in m_data.get("sequences", {}).items()
    }

# 4. Kiểm tra videos_metadata.json
meta_path = os.path.join(WORKSPACE, "backend/data/alphabet_v2/metadata/videos_metadata.json")
video_meta = {}
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        video_meta = json.load(f)

# 5. Kiểm tra file static_landmarks/
static_dir = os.path.join(WORKSPACE, "backend/data/alphabet_v2/static_landmarks")
static_json_files = {}
if os.path.exists(static_dir):
    for f in os.listdir(static_dir):
        if f.endswith(".json"):
            letter = f.replace(".json", "")
            with open(os.path.join(static_dir, f), "r", encoding="utf-8") as fp:
                d = json.load(fp)
            samples = d.get("samples", [])
            stable = [s for s in samples if s.get("phase") == "stable_pose"]
            static_json_files[letter] = {
                "total_samples": len(samples),
                "stable_samples": len(stable),
                "video_id": samples[0].get("video_id") if samples else None
            }

output = {
    "static_npz_classes": classes_static,
    "static_npz_counts": static_counts,
    "vsl_dataset_info": vsl_dataset_info,
    "motion_info": motion_info,
    "static_json_files": static_json_files,
    "video_meta_letters": {k: v.get("video_id") for k, v in video_meta.items()}
}

print(json.dumps(output, indent=2, ensure_ascii=False))

