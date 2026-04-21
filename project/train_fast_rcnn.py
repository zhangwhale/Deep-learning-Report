import torch
import torchvision
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import xml.etree.ElementTree as ET

# =========================
# 1. VOC类别
# =========================
CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

# =========================
# 2. 解析VOC标注
# =========================
def parse_voc(target):
    boxes, labels = [], []
    objs = target["annotation"]["object"]

    if not isinstance(objs, list):
        objs = [objs]

    for obj in objs:
        bbox = obj["bndbox"]

        boxes.append([
            float(bbox["xmin"]),
            float(bbox["ymin"]),
            float(bbox["xmax"]),
            float(bbox["ymax"])
        ])

        labels.append(CLASSES.index(obj["name"]) + 1)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }

# =========================
# 3. Dataset
# =========================
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.dataset = VOCDetection(
            root=root,
            year="2007",
            image_set="trainval",
            download=False
        )
        self.transform = T.ToTensor()

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return self.transform(img), parse_voc(target)

    def __len__(self):
        return len(self.dataset)

# =========================
# 4. collate
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))

# =========================
# 5. 数据集（小规模测试）
# =========================
dataset = VOCDataset("E:/VOCdevkit")
dataset = Subset(dataset, range(300))

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================
# 6. Fast R-CNN（模拟版）
# =========================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# 替换分类头（VOC 21类）
num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# 7. optimizer
# =========================
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# =========================
# 8. 训练
# =========================
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(loader):

        images = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"[Fast R-CNN] Epoch {epoch} Iter {i} Loss {loss.item():.4f}")

    print(f"Epoch {epoch} Avg Loss: {total_loss/len(loader):.4f}")

# =========================
# 9. 保存模型
# =========================
torch.save(model.state_dict(), "fast_rcnn_voc.pth")
print("Fast R-CNN训练完成")