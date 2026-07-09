from ultralytics import YOLO

# Load your custom or pretrained YOLO26 model (e.g., yolo26n, yolo26s)
model = YOLO('model_name.pt') # replace with your model name and location if not in same folder

# Export to OpenVINO format
model.export(format='openvino', imgsz=640)
