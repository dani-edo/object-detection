import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
model = YOLO('./runs/detect/train13_(urine_test_v2_3params_with_200epoch_32batch)/weights/last.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)

# Load the video file
input_video_path = 'source/dipstick-urobilinogen-light.mov'
output_video_path = 'result/train13_(urine_test_v2_3params_with_200epoch_32batch).mov'

# Open the video using OpenCV
video_capture = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize dictionaries to store counts and confidence scores
class_count = defaultdict(int)  # To count occurrences of each class
class_confidences = defaultdict(list)  # To store confidence scores for each class

# temporary EDO
# Define the class to be excluded from analysis
excluded_class = "Dipstick Urine Test Card"

# Iterate over each frame
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break
    
    # Apply YOLOv8 object detection
    results = model(frame)[0]
    
    # Iterate through the detections and draw bounding boxes
    for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[int(cls)]} {conf:.2f}'
        class_name = model.names[int(cls)]

        # temporary EDO
        # Exclude the specified class from being added to analysis data
        if class_name == excluded_class:
            continue
        
        # Draw bounding box and label on the frame
        if conf > 0.5: 
            class_count[class_name] += 1
            class_confidences[class_name].append(conf)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)  # Bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label above the box
    
    # Write the processed frame to the output video
    out_video.write(frame)
    
    # Print progress
    frame_count += 1
    print(f'Processed frame {frame_count}/{total_frames}')

# Release resources
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')

# Find the most frequently detected parameter
most_tracked_parameter = max(class_count, key=class_count.get)
most_tracked_count = class_count[most_tracked_parameter]

# Find the parameter with the highest confidence score
# Ensure that the list is not empty to avoid errors
if class_confidences:
    most_confident_parameter = max(class_confidences, key=lambda x: max(class_confidences[x], default=0))
    highest_confidence_score = max(class_confidences[most_confident_parameter], default=0)
else:
    most_confident_parameter = None
    highest_confidence_score = 0

print(f'Most tracked parameter: {most_tracked_parameter} with {most_tracked_count} occurrences')
print(f'Most confident parameter: {most_confident_parameter} with a highest confidence of {highest_confidence_score:.2f}')
