from ultralytics import YOLO
import sys
import onnx
from onnxsim import simplify

yolo_model = sys.argv[1]

model = YOLO(yolo_model)

onnx_path = model.export(format="onnx", imgsz=(416, 608), dynamic=False, nms=False)

# Simplify the exported ONNX graph before handing it to the Hailo compiler
print("Simplifying ONNX graph...")
onnx_model = onnx.load(onnx_path)

if onnx_model.ir_version > 9:
    print(f"Lowering IR version from {onnx_model.ir_version} to 9")
    onnx_model.ir_version = 9

simplified_model, check = simplify(onnx_model)
if not check:
    raise RuntimeError("onnx-simplifier could not verify the simplified model. Aborting.")

onnx.save(simplified_model, onnx_path)
print(f"Simplified ONNX saved to {onnx_path}")