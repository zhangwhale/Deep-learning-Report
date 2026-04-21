import os
import random
import shutil
import xml.etree.ElementTree as ET

# VOC类别
classes = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

# 路径
voc_path = r"E:\VOCdevkit\VOCdevkit\VOC2007"
output_path = r"E:\VOC_YOLO"

image_dir = os.path.join(voc_path, "JPEGImages")
xml_dir = os.path.join(voc_path, "Annotations")

# 创建目录
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

# 获取所有图片
images = os.listdir(image_dir)
random.shuffle(images)

# 划分比例
train_ratio = 0.8
train_count = int(len(images) * train_ratio)

train_images = images[:train_count]
val_images = images[train_count:]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x*dw, y*dh, w*dw, h*dh)

def convert_annotation(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_file, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )

            bb = convert((w, h), b)
            f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

def process(images, split):
    for img in images:
        img_path = os.path.join(image_dir, img)
        xml_path = os.path.join(xml_dir, img.replace(".jpg", ".xml"))

        # 拷贝图片
        shutil.copy(img_path, os.path.join(output_path, f"images/{split}", img))

        # 转标签
        txt_name = img.replace(".jpg", ".txt")
        txt_path = os.path.join(output_path, f"labels/{split}", txt_name)

        convert_annotation(xml_path, txt_path)

# 执行
process(train_images, "train")
process(val_images, "val")

print("数据集准备完成！")