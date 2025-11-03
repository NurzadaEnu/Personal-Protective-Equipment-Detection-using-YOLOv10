
import re
from pathlib import Path
import pandas as pd
from io import StringIO

def parse_files(results_dir: Path) -> pd.DataFrame:
    frames = []
    for p in results_dir.glob("*.md"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n\s*(?:---+|\#{1,6}\s+.*)\s*\n", txt)
        current_model = None
        for b in blocks:
            m = re.search(r"YOLOv10\s*([nsmblx])", b, flags=re.I)
            if m:
                current_model = f"yolov10{m.group(1).lower()}"
            table = re.search(r"\|(.*?)\|\s*\n(\|[-:]+\|)+\s*\n((\|.*\|\s*\n)+)", b, flags=re.I)
            if table and current_model:
                t = table.group(0)
                t = re.sub(r"mAP@50[\u2013-]95","mAP50_95", t).replace("mAP@50","mAP50")
                df = pd.read_csv(StringIO(t), sep="|", engine="python")
                df = df.drop(columns=[c for c in df.columns if not c.strip()], errors="ignore")
                df.columns = [c.strip() for c in df.columns]
                df = df.applymap(lambda x: str(x).strip())
                if "Class" in df.columns:
                    all_row = df[df["Class"].str.lower()=="all"]
                    if not all_row.empty:
                        r = all_row.iloc[0].to_dict()
                        r.update({"model": current_model})
                        frames.append(r)
    if not frames:
        return pd.DataFrame(columns=["Precision","Recall","mAP50","mAP50_95","model"])
    out = pd.DataFrame(frames).drop_duplicates(subset=["model"])
    for c in ["Precision","Recall","mAP50","mAP50_95"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
