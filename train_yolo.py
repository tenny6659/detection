from ultralytics import YOLO

model = YOLO("./yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=15,        # ✅ balanced
    imgsz=416,
    batch=4,
    workers=0,
    project="runs/cube",
    name="cube_camera"
)
