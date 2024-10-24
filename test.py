from PIL import Image
from io import BytesIO
import requests

response = requests.get("https://images.unsplash.com/photo-1530652101053-8c0db4fbb5de?q=80&w=2787&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
image = Image.open(BytesIO(response.content))

import torchvision

desired_size = (512, 640)
transform = torchvision.transforms.Compose([
 torchvision.transforms.Resize(desired_size),
 torchvision.transforms.ToTensor(),
])
image = transform(image)

from ultralytics import YOLO

model = YOLO('ultralyticsplus/yolov8s')

from easy_explain import YOLOv8LRP

lrp = YOLOv8LRP(model, power=2, eps=1, device='cpu')


explanation_lrp = lrp.explain(image, cls='zebra', contrastive=False).cpu()
lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='Reds', title='Explanation for Class "zebra"')