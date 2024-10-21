import cv2
import torch
import datetime
from collections import defaultdict
from openpyxl import Workbook
import pandas as pd  # For data analysis
from ultralytics import YOLO

# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
model = YOLO('./runs/detect/train13_(urine_test_v2_3params_with_200epoch_32batch)/weights/last.pt')

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

# Create a unique Excel file name using the current date and time
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
xlsx_file_path = f'result/detection_report_{current_time}.xlsx'

# Create a new workbook and select the active worksheet
workbook = Workbook()
sheet = workbook.active
sheet.title = "Detection Results"

# Write the header row in the Excel sheet
sheet.append(['Parameter', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# Dictionary to count detections grouped by parameter (e.g., Urobilinogen)
parameter_counts = defaultdict(int)

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
        class_name = model.names[cls]
        label = f'{class_name} {conf:.2f}'

        # Draw bounding box and label on the frame if confidence is above threshold
        if conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)  # Bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Determine the parameter (e.g., Urobilinogen) from the class name
            parameter = class_name.split(":")[0].strip()  # Extract text before ':'
            
            # Only record data if the class is not "Dipstick Urine Test Card"
            if class_name != "Dipstick Urine Test Card":
                parameter_counts[parameter] += 1  # Increment count for the detected parameter
                
                # Write the data into the Excel sheet
                sheet.append([parameter, class_name, conf, int(x1), int(y1), int(x2), int(y2)])

    # Display the resulting frame
    cv2.imshow('Dipstick Urine Test Live Detection', frame)

    # Press 'q' to break out of the loop and stop the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the workbook to an Excel file
workbook.save(xlsx_file_path)

# Release the camera and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Analyze the most frequent class for each parameter from the saved Excel file
df = pd.read_excel(xlsx_file_path, sheet_name='Detection Results')

# Group by 'Parameter' and 'Class' to count occurrences of each class
class_counts = df.groupby(['Parameter', 'Class']).size().reset_index(name='Count')

# Find the most frequent class for each parameter
most_frequent_classes = class_counts.loc[class_counts.groupby('Parameter')['Count'].idxmax()]

# Print the most frequent classes for each parameter
print("Most frequent classes for each parameter:")
print(most_frequent_classes)

# Optionally, save the analysis results to a new sheet in the same Excel file
with pd.ExcelWriter(xlsx_file_path, mode='a', engine='openpyxl') as writer:
    most_frequent_classes.to_excel(writer, sheet_name='Most Frequent Classes', index=False)

print(f'Analysis data saved to {xlsx_file_path}')
