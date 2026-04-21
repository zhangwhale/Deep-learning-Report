import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["Person", "Car", "Dog", "Cat"]

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    max_samples=200,
    classes=CLASSES
)

# 打乱
dataset = dataset.shuffle(seed=42)

# 划分
train_split = dataset.take(160)
val_split = dataset.exclude([s.id for s in train_split])

# 导出 train
train_split.export(
    export_dir="E:/OpenImages_200",
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
    classes=CLASSES   # ⭐关键
)

# 导出 val
val_split.export(
    export_dir="E:/OpenImages_200",
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
    classes=CLASSES   # ⭐关键
)

print("✔ train + val 导出完成")