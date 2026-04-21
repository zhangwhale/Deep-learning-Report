# ======================
# 1. 导入
# ======================
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os

# ======================
# 2. Dataset（Step2）
# ======================
class OpenImagesDataset(Dataset):

    def __init__(self, img_dir, csv_file):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.image_ids = self.df["ImageID"].unique()
        image_ids = self.df["ImageID"].unique()
        valid_ids = []

        for img_id in image_ids:
            img_path = os.path.join(img_dir, img_id + ".jpg")
            if os.path.exists(img_path):
                valid_ids.append(img_id)

        self.image_ids = valid_ids
        self.df = self.df[self.df["ImageID"].isin(self.image_ids)]
        print("有效图片数量:", len(self.image_ids))
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        while True:  # 关键：循环直到找到正常图片

            img_id = self.image_ids[idx]
            img_path = os.path.join(self.img_dir, img_id + ".jpg")

            img = cv2.imread(img_path)

        # 如果图片读不到 → 跳过
            if img is None:
                idx = (idx + 1) % len(self.image_ids)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img).permute(2,0,1).float() / 255.0

            records = self.df[self.df["ImageID"] == img_id]

            boxes = []
            labels = []

            for _, row in records.iterrows():
                boxes.append([
                    row["XMin"],
                    row["YMin"],
                    row["XMax"],
                    row["YMax"]
                ])
                labels.append(1)

        # ❗如果没有标注，也跳过
            if len(boxes) == 0:
                idx = (idx + 1) % len(self.image_ids)
                continue

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }

            return img, target


# ======================
# 3. 模型（Step3）
# ======================
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

model.to(device)


# ======================
# 4. DataLoader
# ======================
dataset = OpenImagesDataset(
    img_dir=r"E:\OpenImages\train\data",
    csv_file=r"E:\OpenImages\train\labels\detections.csv"
)

loader = DataLoader(    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)


# ======================
# 5. 训练（Step4）
# ======================
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

for epoch in range(5):

    model.train()

    for i, (images, targets) in enumerate(loader):

        images = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch} | Iter {i} | Loss {loss.item():.4f}")

# ======================
# 6. 保存模型
# ======================
save_path = "faster_rcnn_openimages.pth"
torch.save(model.state_dict(),save_path)

print("模型保存路径:", os.path.abspath(save_path))