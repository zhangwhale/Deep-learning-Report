import fiftyone as fo
import fiftyone.zoo as foz

# =========================
# 下载 Open Images V6 子集
# =========================
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],   # 目标检测（bbox）
    classes=["Person", "Car", "Dog"],  # 你要的类别
    max_samples=200,  # 控制大小，避免爆硬盘
    dataset_dir=r"E:\OpenImages"
)

print("下载完成！")