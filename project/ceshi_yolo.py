from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source="E:/VOC_YOLO/images/val",
    save=True
)