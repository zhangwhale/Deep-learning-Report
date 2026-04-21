import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 1. 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

model.load_state_dict(torch.load("faster_rcnn_openimages.pth", map_location=device))
model.to(device)
model.eval()

# 2. 读图
img_path = r"E:\OpenImages\train\data\随便选一张存在的.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = torch.tensor(img_rgb).permute(2,0,1).float()/255
img_tensor = img_tensor.unsqueeze(0).to(device)

# 3. 推理
with torch.no_grad():
    outputs = model(img_tensor)

boxes = outputs[0]['boxes'].cpu().numpy()
scores = outputs[0]['scores'].cpu().numpy()

# 4. 画框
for box, score in zip(boxes, scores):
    if score < 0.5:
        continue

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img_rgb, f"{score:.2f}", (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

plt.imshow(img_rgb)
plt.axis('off')
plt.show()