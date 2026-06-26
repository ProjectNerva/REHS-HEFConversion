from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.export(format="onnx", imgsz=(608, 416), dynamic=True)