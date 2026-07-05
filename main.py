from ultralytics import YOLO
import sys
import onnx
from onnxsim import simplify

yolo_model = sys.argv[1]

# Optional height/width so the export resolution matches the model profile.
# Defaults preserve the original 416x608 behaviour when called with no dims.
height = int(sys.argv[2]) if len(sys.argv) > 2 else 416
width = int(sys.argv[3]) if len(sys.argv) > 3 else 608

model = YOLO(yolo_model)

onnx_path = model.export(format="onnx", imgsz=(height, width), dynamic=False, nms=False)

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