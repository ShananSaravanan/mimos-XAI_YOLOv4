import cv2 as cv
import time
import numpy as np

# Set confidence and NMS thresholds
Conf_threshold = 0.4
NMS_threshold = 0.4

# Define colors for bounding boxes
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Load class names from file
class_name = []
with open('idealv2.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLOv4 model
net = cv.dnn.readNet('idealv2_mimmos.weights', 'idealv2_mimmos.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Create a detection model
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load an image for detection
image_path = "fail(iv2).jpg"  # Replace with your image path
frame = cv.imread(image_path)

# Start timer for FPS calculation
starting_time = time.time()

# Perform detection
classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

# Draw bounding boxes and labels
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = f"{class_name[classid]} : {score:.2f}"
    cv.rectangle(frame, box, color, 2)
    cv.putText(frame, label, (box[0], box[1] - 10),
                cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

# Calculate FPS
endingTime = time.time() - starting_time
fps = len(classes) / endingTime if len(classes) > 0 else 0
cv.putText(frame, f'FPS: {fps:.2f}', (20, 50),
           cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

# Show the output image
cv.imshow('Detections', frame)
cv.waitKey(0)
cv.destroyAllWindows()
