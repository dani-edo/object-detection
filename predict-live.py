import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
# model = YOLO('./runs/detect/train4/weights/last.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)
# model = YOLO('./runs/detect/train10_(urine_test_urobilinogen_with_30epoch)/weights/last.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)
model = YOLO('./runs/detect/train11_(urine_test_urobilinogen_with_100epoch)/weights/last.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)

# Open the webcam using OpenCV (0 is typically the default camera)
video_capture = cv2.VideoCapture(1)

# Check if the webcam opened successfully
if not video_capture.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Get camera properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Iterate over each frame from the camera
while True:
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Apply YOLOv8 object detection
    results = model(frame)[0]
    
    # Iterate through the detections and draw bounding boxes
    for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[cls]} {conf:.2f}'
        
        # Draw bounding box and label on the frame if confidence is above threshold
        if conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)  # Bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('YOLOv8 Live Detection', frame)
    
    # Press 'q' to break out of the loop and stop the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
