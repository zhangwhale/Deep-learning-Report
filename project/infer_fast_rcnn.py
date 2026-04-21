import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# =========================
# 1. 类别（必须和训练一致）
# =========================
CLASSES = [
    "__background__", "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow","diningtable","dog","horse",
    "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

# =========================
# 2. 加载模型
# =========================
num_classes = 21

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# ⭐ 加载你的模型
model.load_state_dict(torch.load(
    "E:/Deeplearning-2/project/fast_rcnn_voc.pth",
    map_location="cpu"
))

model.eval()

# =========================
# 3. 读测试图片（换成你自己的）
# =========================
img_path = r"E:\VOCdevkit\VOCdevkit\VOC2007\JPEGImages\008315.jpg"

image = Image.open(img_path).convert("RGB")

transform = T.ToTensor()
img_tensor = transform(image)

# =========================
# 4. 推理
# =========================
with torch.no_grad():
    output = model([img_tensor])[0]

# =========================
# 5. 可视化
# =========================
plt.imshow(image)

for box, label, score in zip(
    output["boxes"], output["labels"], output["scores"]
):
    if score > 0.5:
        x1, y1, x2, y2 = box.numpy()

        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color='red',
                linewidth=2
            )
        )

        plt.text(
            x1,
            y1,
            f"{CLASSES[label]} {score:.2f}",
            color="yellow",
            fontsize=8
        )

plt.title("Fast R-CNN Detection Result")
plt.axis("off")
plt.show()