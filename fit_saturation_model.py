
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils.parse_md_results import parse_files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--assume_counts", action="append", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_files(Path(args.results_dir))
    if df.empty:
        raise SystemExit("No metrics parsed.")

    # Minimal dataset mapping from flags
    meta = {}
    for spec in args.assume_counts:
        name, rest = spec.split(":",1)
        fields = {"dataset": name.strip()}
        for token in rest.split(","):
            k,v = token.split("=")
            fields[k.strip()] = int(v.strip())
        meta[fields["dataset"]] = fields
    meta_df = pd.DataFrame(meta.values()) if meta else pd.DataFrame()

    # Expand per-model rows across datasets if needed (visualization only)
    # Here we just attach the first dataset sizes per convenience to build V terms
    if not meta_df.empty:
        df["dataset"] = meta_df.iloc[0]["dataset"]
        df["images"]  = meta_df.iloc[0]["images"]
    else:
        df["dataset"] = "UNKNOWN"
        df["images"]  = 1000

    df["V"] = df["images"].astype(float)
    df["V2"] = df["V"]**2
    df["D"] = 1.0

    for letter in ["s","m","b","l","x"]:
        df[f"C_{letter}"] = (df["model"]==f"yolov10{letter}").astype(int)

    y = df["mAP50"].astype(float)
    X = pd.DataFrame({
        "const": 1.0, "V": df["V"], "V2": df["V2"], "D": df["D"],
        "C1": df["C_s"], "C2": df["C_m"], "C3": df["C_b"], "C4": df["C_l"], "C5": df["C_x"]
    })
    model = sm.OLS(y, X, missing="drop").fit()
    (out_dir/"model_summary.txt").write_text(model.summary().as_text())
    (out_dir/"parsed_metrics.csv").write_text(df.to_csv(index=False))

    # Plot
    fig_path = out_dir/"figure4_mAP_vs_size.png"
    plt.figure()
    plt.scatter(df["V"], df["mAP50"], label="mAP@50")
    z = np.polyfit(df["V"], df["mAP50"], deg=2)
    xs = np.linspace(df["V"].min(), df["V"].max(), 100)
    ys = z[0]*xs**2 + z[1]*xs + z[2]
    plt.plot(xs, ys, label="poly fit")
    plt.xlabel("Dataset size (images)"); plt.ylabel("mAP@50"); plt.title("Impact of Dataset Size")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path, dpi=220)
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    main()
