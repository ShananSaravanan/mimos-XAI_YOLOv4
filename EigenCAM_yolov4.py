import cv2 as cv
import time
import numpy as np
import warnings
import torch    
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from tool import darknet2pytorch
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Set confidence and NMS thresholds
Conf_threshold = 0.4
NMS_threshold = 0.4

# Define colors for bounding boxes
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Load class names from file
model_name = "idealv4_mimmos"  # Use distinct variable name (for dynamic uses can be modified later on through GUI variables)
class_name = []
with open(f'{model_name}.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLOv4 model
net = cv.dnn.readNet(f'{model_name}.weights', f'{model_name}.cfg') #opencv method to make use of weights and cfg file into one model 
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Create a detection model
detection_model = cv.dnn_DetectionModel(net)  # model to be used (for detection)
detection_model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load and preprocess the image
image_path = "fail(iv4).jpg"  # test image (can be dynamically adjusted through GUI later on)
img = np.array(Image.open(image_path))
rgb_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # Convert from RGB to BGR (OpenCV requirements)

# Start timer for FPS calculation
starting_time = time.time()

# Perform detection
classes, scores, boxes = detection_model.detect(rgb_img, Conf_threshold, NMS_threshold) 

# Draw bounding boxes and labels on the image (based on the score classes retrieved from detection)
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = f"{class_name[classid]} : {score:.2f}"
    cv.rectangle(rgb_img, box, color, 2)
    cv.putText(rgb_img, label, (box[0], box[1] - 10),
                cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

#To Resolve Idealv2 multiple bounding box detection problem
# # If there are detections, proceed to select the one with the highest score
# if len(classes) > 0:
#     # Find the index of the detection with the highest score
#     max_index = np.argmax(scores)
    
#     # Get the highest score detection details
#     classid = classes[max_index]
#     score = scores[max_index]
#     box = boxes[max_index]
    
#     # Define the color for this class ID
#     color = COLORS[int(classid) % len(COLORS)]
#     label = f"{class_name[classid]} : {score:.2f}"
    
#     # Draw the bounding box and label for the highest scoring detection
#     cv.rectangle(rgb_img, box, color, 2)
#     cv.putText(rgb_img, label, (box[0], box[1] - 10),
#                cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

# Calculate FPS
endingTime = time.time() - starting_time
fps = 1 / endingTime if len(classes) > 0 else 0
cv.putText(rgb_img, f'Detector Time: {fps:.2f}', (20, 50),
           cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) #write the stats on image


# Load and preprocess the image for EigenCAM
img_resized = cv.resize(img, (416, 416))  # Resize to match your model's input size (416x416x3)
img_resized = np.float32(img_resized) / 255  # Normalize to [0, 1]
transform = transforms.ToTensor()
tensor = transform(img_resized).unsqueeze(0)

# Load your YOLOv4 model for CAM
cam_model = darknet2pytorch.Darknet(f'{model_name}.cfg', inference=True) #darknet2pytorch conversion (to be used for XAI purposes as it only supports pytorch format to evaluate)
cam_model.load_weights(f'{model_name}.weights')
cam_model.eval()

# Start timer for XAI work
xai_start_time = time.time()

# Forward pass to get detections for CAM
with torch.no_grad():
    outputs = cam_model(tensor)
    detections = outputs[0]

# Parse detections
def parse_detections(detections):
    boxes, colors, names = [], [], []
    for detection in detections[0]:
        detection_np = detection.cpu().numpy() if torch.is_tensor(detection) else detection
        if detection_np.shape[0] < 4:
            continue
        xmin = int(detection_np[0])
        ymin = int(detection_np[1])
        xmax = int(detection_np[2])
        ymax = int(detection_np[3])
        confidence = 1.0  # Placeholder
        class_id = 2  # Placeholder
        color = COLORS[class_id]
        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(class_name[class_id])  # Adjust as necessary
    return boxes, colors, names

# Get boxes, colors, and names
boxes, colors, names = parse_detections(detections)

# To inspect the model layers to see available options
# for idx, layer in enumerate(cam_model.models):
#     print(f"Layer {idx}: {layer}")


# Set target layers for EigenCAM
target_layers = [cam_model.models[149].conv102]  # retrieved based on examining the layers of the model (last convolutional layers are usually chosen)

#EigenCAM docs
# Initialize EigenCAM
cam = EigenCAM(cam_model, target_layers)

# Get the index of the predicted class
predicted_class = torch.argmax(outputs[1]).item()

# Generate CAM
grayscale_cam = cam(tensor, targets=[predicted_class])[0, :, :]

# Show CAM on the original image
cam_image = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

# Function to renormalize CAM within bounding boxes
def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv.rectangle(eigencam_image_renormalized, (xmin, ymin), (xmax, ymax), color, 2)
        cv.putText(eigencam_image_renormalized, str(name), (xmin, ymin - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return eigencam_image_renormalized

# Generate renormalized CAM image
renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img_resized, grayscale_cam)

# Stop timer for XAI work and calculate elapsed time
xai_ending_time = time.time() - xai_start_time

cv.putText(cam_image, f'XAI Time: {xai_ending_time:.2f}', (20, 50),
           cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) #apply stats on image

# Resize the image
rgb_img = cv.resize(rgb_img, (416, 416))  # based on image conventional size (416x416)

# Combine the images horizontally
combined_image = np.hstack((rgb_img, cam_image))

# Convert the combined image to the appropriate format for Matplotlib
combined_image = combined_image.astype(np.uint8)

# Display the combined image
plt.imshow(combined_image)
plt.axis('off')  # Turn off axis labels
plt.show()
