from ultralytics import YOLO

config_path = 'config.yaml'

# Load a model
model = YOLO('yolov8n.pt')  # load pre trained model

# Use the model
model.train(data=config_path, epochs=100, batch=16)  # train the model (change epochs to 200 and batch to 32 for ideal condition)