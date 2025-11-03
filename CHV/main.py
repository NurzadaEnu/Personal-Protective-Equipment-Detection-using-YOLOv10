#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def auto_device_default() -> str:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return ",".join(str(i) for i in range(n)) if n > 0 else "0"
    return "cpu"


def auto_workers_default(cap=60) -> int:
    cpu_cnt = os.cpu_count() or 8
    return max(4, min(cap, cpu_cnt - 1))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Ultralytics YOLO with simple CLI (model + data)."
    )
    # Positional
    p.add_argument("model", help="Model config/weights (e.g., yolo11m.yaml or yolo11m.pt)")
    p.add_argument("data", help="Dataset YAML path (e.g., /path/to/data.yaml)")

    # Common training knobs (defaults match your snippet)
    p.add_argument("--epochs", "-e", type=int, default=250)
    p.add_argument("--imgsz", "-s", type=int, default=640)
    p.add_argument("--batch", "-b", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default=auto_device_default(),
                   help="GPU ids as '0,1,2,3' or 'cpu'. Default auto-detects all GPUs.")
    p.add_argument("--workers", type=int, default=auto_workers_default())
    p.add_argument("--project", default="./runs")
    p.add_argument("--name", default=None, help="Run name under project dir.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    return p.parse_args()


def main():
    args = parse_args()

    # Derive a friendly default run name if none provided
    if args.name is None:
        args.name = f"{Path(args.model).stem}-exp"

    # Instantiate model (YAML => fresh model, .pt/.ptn/.onnx => weights)
    model = YOLO(args.model)

    # Ultralytics expects device as string like "0,1,2,3" or "cpu"
    device_str = args.device if isinstance(args.device, str) else str(args.device)

    # Kick off training
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device_str,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        optimizer=args.optimizer,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
