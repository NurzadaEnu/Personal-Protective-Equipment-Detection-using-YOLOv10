import os, glob
import numpy as np
import pandas as pd

def count_yolo_objects(labels_dir: str, num_classes: int):
    """
    YOLO labels: each line = "class x y w h"
    Returns:
      total_objects, class_counts, D_maxmin_nonzero, D_95_5 (robust)
    """
    class_counts = np.zeros(num_classes, dtype=int)
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    for fp in label_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                if 0 <= cls < num_classes:
                    class_counts[cls] += 1

    total_objects = int(class_counts.sum())

    nonzero = class_counts[class_counts > 0]
    if len(nonzero) == 0:
        D = np.nan
        D_95_5 = np.nan
    else:
        D = float(nonzero.max() / nonzero.min())
        p95 = np.percentile(nonzero, 95)
        p5 = np.percentile(nonzero, 5)
        D_95_5 = float(p95 / p5) if p5 > 0 else np.nan

    return total_objects, class_counts, D, D_95_5


# ---- Put YOUR paths here ----
DATASETS = {
    "SH17":   {"labels_dir": "/path/to/SH17/train/labels",   "images": 8099, "num_classes": 17},
    "CHV":    {"labels_dir": "/path/to/CHV/train/labels",    "images": 1330, "num_classes": 6},
    "SHEL5K": {"labels_dir": "/path/to/CHV/train/labels", "images": 5000, "num_classes": 4},
}

rows = []
for name, info in DATASETS.items():
    V_objects, counts, D, D_95_5 = count_yolo_objects(info["labels_dir"], info["num_classes"])
    rows.append({
        "dataset": name,
        "num_classes": info["num_classes"],
        "images": info["images"],
        "V_objects": V_objects,
        "D_maxmin": D,
        "D_95_5": D_95_5,
    })

df_stats = pd.DataFrame(rows)
print(df_stats)
