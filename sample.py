from ultralytics import YOLO
import os

# Load fresh model
model = YOLO('yolo11m.yaml') # yolo model type [n,m,l,s]

yolo_dir = '/path_to/chv'
model.train(
    data=os.path.join(yolo_dir, 'data.yaml'),
    epochs=250,
    imgsz=640,
    batch=16,
    patience=20,
    device=[0,1,2,3],
    workers=60,
    project='./runs',
    seed=42,
    optimizer='AdamW',
    amp=True
)
