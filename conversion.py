import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
import onnx
from onnx2pytorch import ConvertModel

# Convert ONNX model to PyTorch model
onnx_model_path = 'yolov4_1_3_416_416_static.onnx'
onnx_model = onnx.load(onnx_model_path)
pytorch_model = ConvertModel(onnx_model)



# Ensure gradients are computed
for param in pytorch_model.parameters():
    param.requires_grad = True

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_tensor.requires_grad = True  # Ensure tensor requires gradient
    return img_tensor

image_path = 'test.jpg'
input_tensor = preprocess_image(image_path)

# Define GradCAM with a different layer name if needed
target_layer_name = 'Conv_/models.145/conv98/Conv_output_0'  # Try a different layer
cam_extractor = GradCAM(pytorch_model, target_layer=target_layer_name)

# Perform a forward pass with gradients enabled
pytorch_model.eval()  # Set to evaluation mode
output = pytorch_model(input_tensor)

print(f"Input Tensor: {input_tensor}")
print(f"Input Tensor Shape: {input_tensor.shape}")



# Print the shape and structure of the output to see what's being returned
print(f"Model output: {output}")
print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output if isinstance(o, torch.Tensor)]}")

for name, param in pytorch_model.named_parameters():
    print(f"Layer {name}: NaN count: {torch.isnan(param).sum()}")


# If the output is a list, extract the relevant part
if isinstance(output, list):
    # Example: If you expect the first item to be the predictions, modify as needed
    output = output[0]

# Now choose the target class for GradCAM
target_class = output.argmax(dim=1).item()

# Extract GradCAM
cam = cam_extractor(target_class, input_tensor)

# Plot the result
plt.imshow(cam[0].cpu().numpy(), cmap='jet', alpha=0.5)
plt.show()


def overlay_cam_on_image(img_path, cam):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((416, 416))  # Resize to match model input dimensions

    # Convert CAM to numpy array and normalize
    cam = cam[0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    # Convert CAM to PIL image
    cam_image = Image.fromarray((cam * 255).astype(np.uint8), mode='L')
    cam_image = cam_image.convert('RGB')  # Convert to RGB mode

    # Overlay CAM on image
    blended_image = Image.blend(img, cam_image, alpha=0.5)
    blended_image.show()

# Use the function to overlay CAM on the original image
overlay_cam_on_image(image_path, cam)
