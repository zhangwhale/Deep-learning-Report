from ultralytics import YOLO

# 1. 加载模型（预训练）
model = YOLO("yolov8n.pt")

# 2. 开始训练
model.train(
    data="E:/VOC_YOLO/voc.yaml",  # 数据集配置文件
    epochs=10,                   # 先跑10轮测试
    imgsz=640,                   # 图片尺寸
    batch=8                      # 批次
)

print("训练完成！")