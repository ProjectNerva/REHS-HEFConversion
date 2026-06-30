from ultralytics import YOLO
import sys

yolo_model = sys.argv[1]
height = int(sys.argv[2])
width = int(sys.argv[3])

model = YOLO(yolo_model)

model.export(format="onnx", imgsz=(height, width), dynamic=True)