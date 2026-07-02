import sys
import json
import numpy as np
import onnx
from onnx import shape_inference as onnx_shape_inference
from hailo_sdk_client import ClientRunner

model_name = sys.argv[1]
onnx_path = sys.argv[2]
# Load your custom preprocessing calibration data (1000 - 2000 real data samples)
# Formatted as a numpy array, matching your model's exact input shape
calibration_data = np.load(sys.argv[3])

# Fetch start/end node names from the (already simplified) ONNX graph
onnx_model = onnx.load(onnx_path)
input_initializers = {node.name for node in onnx_model.graph.initializer}
start_node_names = [node.name for node in onnx_model.graph.input if node.name not in input_initializers]

# YOLO26's NMS-free end-to-end Detect head bakes a TopK/GatherElements selection
# block directly after the raw per-scale box/class Conv outputs, which Hailo can't
# compile on-chip. Cutting at a single downstream Transpose parsed/quantized fine,
# but failed to compile ("No valid partition found") - ending at these six raw Conv
# outputs instead (Hailo's own suggested NMS-aware end nodes) is what actually
# allocates successfully on Hailo-8.
END_NODE_NAMES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",
]
graph_node_names = {node.name for node in onnx_model.graph.node}
if all(name in graph_node_names for name in END_NODE_NAMES):
    end_node_names = END_NODE_NAMES
else:
    end_node_names = [node.name for node in onnx_model.graph.output]

print("Start nodes:", start_node_names)
print("End nodes:  ", end_node_names)

# Auto-detect num_classes and regression_length from the ONNX Conv end nodes.
# cv2 branches (box regression) output 4 * regression_length channels.
# cv3 branches (class prediction) output num_classes channels.
# ONNX Conv outputs are NCHW, so channels is dim[1] of the output tensor.
inferred = onnx_shape_inference.infer_shapes(onnx_model)
shape_map = {}
for v in list(inferred.graph.value_info) + list(inferred.graph.output):
    dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
    shape_map[v.name] = dims

cv2_ch = None
cv3_ch = None
for node in inferred.graph.node:
    if node.name not in end_node_names or not node.output:
        continue
    dims = shape_map.get(node.output[0])
    if not dims or len(dims) < 2:
        continue
    ch = dims[1]  # NCHW: index 1 is channels
    if "cv2" in node.name and cv2_ch is None:
        cv2_ch = ch
    elif "cv3" in node.name and cv3_ch is None:
        cv3_ch = ch

if cv2_ch is None or cv3_ch is None:
    print(f"ERROR: Auto-detection failed (cv2_ch={cv2_ch}, cv3_ch={cv3_ch}). Check that END_NODE_NAMES match the ONNX graph.", file=sys.stderr)
    sys.exit(1)

num_classes = cv3_ch
regression_length = cv2_ch // 4
print(f"Auto-detected: num_classes={num_classes}, regression_length={regression_length} ({'DFL-free' if regression_length == 1 else 'DFL'})")

# Write the NMS config JSON so model_optimization_nms.alls can reference it.
# Thresholds are baked into the HEF at compile time and cannot be changed at runtime.
nms_config = {
    "classes": num_classes,
    "regression_length": regression_length,
    "anchors": {
        "strides": [8, 16, 32],
        "sizes": [[1, 1], [1, 1], [1, 1]],
        "scale_factors": [0.5, 0.5],
    },
    "nms_iou_thresh": 0.45,
    "score_threshold": 0.25,
    "nms_max_output_per_class": 300,
    "post_nms_topk": 300,
}
with open("yolo_nms_config.json", "w") as f:
    json.dump(nms_config, f, indent=2)
print("NMS config written to yolo_nms_config.json")

# hw_arch is fixed here (not at compile time) and carries through the saved HAR files.
# Options: 'hailo8', 'hailo8l', 'hailo15', 'hailo15l'
runner = ClientRunner(hw_arch="hailo8")

hn, npz = runner.translate_onnx_model(
    onnx_path,
    model_name,
    start_node_names=start_node_names,
    end_node_names=end_node_names,
)

har_path = f"{model_name}_fp32.har"
runner.save_har(har_path)
print("Parsing complete. HAR saved.")

# Load the previously saved HAR archive
runner = ClientRunner(har=f"{model_name}_fp32.har")

# model_optimization_nms.alls is identical to model_optimization.alls but adds
# nms_postprocess(), which appends Hailo's NMS engine as a layer inside the HEF.
runner.load_model_script("model_optimization_nms.alls")

# Run full quantization algorithm using real data to minimize math accuracy loss
runner.optimize(calibration_data)

# Save the quantized model archive
quantized_har_path = f"{model_name}_quantized.har"
runner.save_har(quantized_har_path)
print("Quantization complete. Quantized HAR saved.")

# Load the quantized archive
runner = ClientRunner(har=f"{model_name}_quantized.har")

# Compile targeting the hardware architecture fixed when the runner was first
# created (hailo8) - carried through from the saved HAR file
hef = runner.compile()

# Write out the final binary deployable file
hef_output_path = f"{model_name}_nms.hef"
with open(hef_output_path, "wb") as f:
    f.write(hef)

print(f"Success! NMS-baked model compiled into: {hef_output_path}")
print("Use hef_infer_nms.py for inference — no host-side decode or NMS needed.")
