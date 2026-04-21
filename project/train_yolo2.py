from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="E:/OpenImages_200/dataset.yaml",
    epochs=10,
    imgsz=640,
    batch=8
)