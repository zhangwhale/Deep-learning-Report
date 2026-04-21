import torch
import torchvision
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

# ========================
# 1. 类别
# ========================
CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

# ========================
# 2. 标注解析
# ========================
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

# ========================
# 3. Dataset
# ========================
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.dataset = VOCDetection(root=root, year="2007", image_set="trainval", download=False)
        self.transforms = T.ToTensor()

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return self.transforms(img), parse_voc(target)

    def __len__(self):
        return len(self.dataset)

# ========================
# 4. DataLoader
# ========================
def collate_fn(batch):
    return tuple(zip(*batch))

dataset = VOCDataset("E:/VOCdevkit")

# ⭐ 关键：先用小数据调试
dataset = Subset(dataset, range(200))

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

# ========================
# 5. 模型（推荐用预训练）
# ========================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

# 改类别数
num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

# ========================
# 6. 设备
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========================
# 7. 优化器
# ========================
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# ========================
# 8. 训练
# ========================
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(data_loader):
        print(f"Epoch {epoch+1} | batch {i}")

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "faster_rcnn_voc_v2.pth")
print("训练完成")