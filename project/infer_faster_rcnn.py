import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# =========================
# 1. VOC类别（必须和训练一致）
# =========================
CLASSES = [
    "__background__",
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

# =========================
# 2. 加载模型（结构必须一致！）
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

# 加载训练好的权重
model.load_state_dict(
    torch.load("faster_rcnn_voc_v2.pth", map_location=device)
)

model.to(device)
model.eval()

# =========================
# 3. 读取测试图片
# =========================
img_path = r"E:\VOCdevkit\VOCdevkit\VOC2007\JPEGImages\008315.jpg"

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = F.to_tensor(img_rgb).to(device)

# =========================
# 4. 推理
# =========================
with torch.no_grad():
    output = model([img_tensor])[0]

# =========================
# 5. 可视化结果
# =========================
plt.figure(figsize=(10,8))
plt.imshow(img_rgb)

boxes = output["boxes"].cpu()
labels = output["labels"].cpu()
scores = output["scores"].cpu()

for box, label, score in zip(boxes, labels, scores):
    if score < 0.5:
        continue

    x1, y1, x2, y2 = box

    cls_name = CLASSES[label]

    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
    )

    plt.text(
        x1, y1,
        f"{cls_name}:{score:.2f}",
        color="yellow",
        fontsize=8,
        bbox=dict(facecolor="red", alpha=0.5)
    )

plt.title("Faster R-CNN VOC Detection Result")
plt.axis("off")
plt.show()
