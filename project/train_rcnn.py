import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import selectivesearch
from torchvision import transforms

# =========================
# 1. VOC类别（可选）
# =========================
CLASSES = [
    "__background__", "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow","diningtable","dog","horse",
    "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

# =========================
# 2. 读取图片（VOC）
# =========================
img_path = r"E:\VOCdevkit\VOCdevkit\VOC2007\JPEGImages\008315.jpg"

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# 3. Selective Search生成候选框
# =========================
_, regions = selectivesearch.selective_search(
    img_rgb,
    scale=500,
    sigma=0.9,
    min_size=10
)

boxes = []
for r in regions:
    x, y, w, h = r['rect']

    if w < 30 or h < 30:
        continue

    boxes.append((x, y, w, h))

print("候选框数量:", len(boxes))

# =========================
# 4. CNN特征提取（ResNet）
# =========================
device = torch.device("cpu")

model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# 5. R-CNN推理（逐框）
# =========================
results = []

for i, (x, y, w, h) in enumerate(boxes[:50]):

    crop = img_rgb[y:y+h, x:x+w]

    try:
        tensor = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(tensor)

        score = torch.norm(feat).item()

        # 简单阈值（模拟分类）
        if score > 10:
            results.append((x, y, w, h, score))

    except:
        continue

print("检测框数量:", len(results))

# =========================
# 6. 可视化结果
# =========================
plt.figure(figsize=(10,8))
plt.imshow(img_rgb)

for x, y, w, h, score in results:
    plt.gca().add_patch(
        plt.Rectangle(
            (x, y), w, h,
            fill=False,
            color='red',
            linewidth=2
        )
    )

plt.title("R-CNN Detection Result (Simplified)")
plt.axis("off")
plt.show()