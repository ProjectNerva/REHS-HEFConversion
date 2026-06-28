from ultralytics import YOLO
import sys

yolo_model = sys.argv[1]

model = YOLO(yolo_model)

model.export(format="onnx", imgsz=(608, 416), dynamic=True)