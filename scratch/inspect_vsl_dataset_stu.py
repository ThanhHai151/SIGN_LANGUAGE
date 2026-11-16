import numpy as np

ds1 = np.load('backend/data/vsl_dataset.npz', allow_pickle=True)
classes1 = list(ds1['classes'])
X1 = ds1['X']
y1 = ds1['y']

for let in ['T', 'U', 'S']:
    idx = classes1.index(let)
    x_cls = X1[y1 == idx]
    print(f"\n=== vsl_dataset.npz for {let} (shape {x_cls.shape}) ===")
    # Recall the 72 features:
    # 0..62: normalized 21 coords (x,y,z) / palm_scale
    # 63: d(4,8) / ps (d_TI)
    # 64: d(8,12) / ps (d_IM)
    # 65: d(4,12) / ps (d_TM)
    # 66: d(4,20) / ps (d_TP)
    # 67: thumb_ext
    # 68: index_ext
    # 69: middle_ext
    # 70: ring_ext
    # 71: pinky_ext
    print("Extra features (63..71):")
    print("d_TI (mean +/- std):", np.mean(x_cls[:, 63]), "+/-", np.std(x_cls[:, 63]))
    print("d_IM (mean +/- std):", np.mean(x_cls[:, 64]), "+/-", np.std(x_cls[:, 64]))
    print("d_TM (mean +/- std):", np.mean(x_cls[:, 65]), "+/-", np.std(x_cls[:, 65]))
    print("d_TP (mean +/- std):", np.mean(x_cls[:, 66]), "+/-", np.std(x_cls[:, 66]))
    print("thumb_ext:", np.mean(x_cls[:, 67]))
    print("index_ext:", np.mean(x_cls[:, 68]))
    print("middle_ext:", np.mean(x_cls[:, 69]))
    print("ring_ext:", np.mean(x_cls[:, 70]))
    print("pinky_ext:", np.mean(x_cls[:, 71]))

