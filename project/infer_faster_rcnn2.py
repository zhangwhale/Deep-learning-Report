import torch
import torchvision
import cv2
import matplotlib.pyplot as plt

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# =========================
# 1. 类别（必须和训练一致）
# =========================
CLASSES = ["__background__", "Person", "Car", "Dog", "Cat"]

# =========================
# 2. 加载模型
# =========================
num_classes = 5

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(
    "E:/Deeplearning-2/project/faster_rcnn_openimages.pth",
    map_location="cpu"
))

model.eval()

# =========================
# 3. 测试图片（随便选一张）
# =========================
img_path = r"E:\OpenImages_200\images\val\xxx.jpg"  # ⭐改成你的

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = torch.tensor(img_rgb / 255., dtype=torch.float32).permute(2,0,1)

# =========================
# 4. 推理
# =========================
with torch.no_grad():
    output = model([img_tensor])[0]

# =========================
# 5. 可视化
# =========================
plt.imshow(img_rgb)

for box, label, score in zip(
    output["boxes"], output["labels"], output["scores"]
):
    if score > 0.4:   # ⭐阈值可调
        x1, y1, x2, y2 = box.numpy()

        plt.gca().add_patch(
            plt.Rectangle((x1, y1),
                          x2 - x1,
                          y2 - y1,
                          fill=False,
                          linewidth=2)
        )

        plt.text(
            x1, y1,
            f"{CLASSES[label]} {score:.2f}",
            fontsize=8
        )

plt.title("Faster R-CNN (OpenImages)")
plt.axis("off")
plt.show()