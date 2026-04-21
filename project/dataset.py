from torch.utils.data import Dataset
import pandas as pd
import torch
import cv2
import os

class OpenImagesDataset(Dataset):

    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        self.images = self.df["ImageID"].unique()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_id = self.images[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
            labels.append(1)  # 简化：单类 or 自己映射

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            img = self.transform(img)

        return img, target