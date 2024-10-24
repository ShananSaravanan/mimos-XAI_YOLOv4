import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the YOLO model in PyTorch
model = torch.load('path_to_yolov4_model.pt', map_location='cpu')  # Ensure you're using the converted PyTorch model

# Set the model to evaluation mode
model.eval()

# Load and preprocess the image
img_path = 'path_to_image.jpg'
img = Image.open(img_path)
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Apply the GradCAM
target_layer = 'model.22'  # You may need to inspect the model and adjust this based on YOLOv4 layer numbering
cam_extractor = GradCAM(model, target_layer)

# Get the model's output
with torch.no_grad():
    output = model(img_tensor)

# Choose the class index for the object you want to explain
class_idx = output[0].argmax(dim=1).item()  # Assuming output is a detection, you can adjust this logic

# Compute the GradCAM
cam = cam_extractor(class_idx, output)

# Visualize the CAM on the image
result = overlay_mask(np.array(img), cam.squeeze().numpy(), alpha=0.5)

# Plot the result
plt.imshow(result)
plt.axis('off')
plt.show()
