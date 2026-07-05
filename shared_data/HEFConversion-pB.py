import sys
import json
import numpy as np
import onnx
import yaml
from onnx import shape_inference as onnx_shape_inference
from hailo_sdk_client import ClientRunner

model_name = sys.argv[1]
onnx_path = sys.argv[2]
# Load your custom preprocessing calibration data (1000 - 2000 real data samples)
# Formatted as a numpy array, matching your model's exact input shape
calibration_data = np.load(sys.argv[3])

# Optional model profile (YAML). When absent, the original YOLO26 defaults below
# are preserved, so running without a config behaves exactly as before.
config = {}
if len(sys.argv) > 4 and sys.argv[4]:
    with open(sys.argv[4]) as f:
        config = yaml.safe_load(f) or {}
parse_cfg = config.get("parse", {}) or {}
hw_arch = (config.get("hailo", {}) or {}).get("hw_arch", "hailo8")

# Fetch start/end node names from the (already simplified) ONNX graph
onnx_model = onnx.load(onnx_path)
input_initializers = {node.name for node in onnx_model.graph.initializer}

# start_nodes: explicit list from the profile, else derive from the graph inputs.
cfg_start = parse_cfg.get("start_nodes", "auto")
if isinstance(cfg_start, list) and cfg_start:
    start_node_names = cfg_start
else:
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
graph_output_names = [node.name for node in onnx_model.graph.output]

# end_nodes precedence:
#   1. explicit list in the profile  -> use it (must exist in the graph)
#   2. no/"auto" config, YOLO26 head present -> the 6 hardcoded Conv outputs
#   3. otherwise -> the ONNX graph outputs
cfg_end = parse_cfg.get("end_nodes", "auto")
if isinstance(cfg_end, list) and cfg_end:
    missing = [n for n in cfg_end if n not in graph_node_names]
    if missing:
        print(f"ERROR: config end_nodes not found in ONNX graph: {missing}", file=sys.stderr)
        sys.exit(1)
    end_node_names = cfg_end
elif all(name in graph_node_names for name in END_NODE_NAMES):
    end_node_names = END_NODE_NAMES
else:
    end_node_names = graph_output_names

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

# Detect the square input resolution (H, W) from the ONNX input tensor (NCHW).
# Falls back to 640 for dynamic/unspecified dims.
input_dims = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
img_h = input_dims[2] if len(input_dims) >= 4 and input_dims[2] > 0 else 640
img_w = input_dims[3] if len(input_dims) >= 4 and input_dims[3] > 0 else 640

# Pick Hailo's decoder template from the detected head type. This is critical:
# yolov8's decoder applies DFL (softmax-expectation over regression_length bins).
# On a DFL-free head (regression_length=1) that expectation is over a single bin,
# which is identically 0 — so every box collapses to a zero-size point at its cell
# center. YOLO26 / YOLOv10 are DFL-free and must use yolov6 (direct distance decode);
# YOLOv8 / YOLO11 are DFL and must use yolov8. Mirrors Hailo's default_nms_config_*.json.
# engine is coupled to meta_arch: Hailo runs yolov6 NMS on the neural core only
# (nn_core) and yolov8 NMS on the host only (cpu). Picking the wrong one raises
# UnsupportedMetaArchError ("cannot be run on host" / "on chip").
if regression_length == 1:
    meta_arch = "yolov6"          # direct l,t,r,b distance decode (no DFL)
    decoder_prefix = "fused_bbox_decoder"
    engine = "nn_core"            # yolov6 NMS runs on the Hailo chip
else:
    meta_arch = "yolov8"          # DFL decode over regression_length bins
    decoder_prefix = "bbox_decoder"
    engine = "cpu"                # yolov8 NMS runs on the deployment host CPU
print(f"Selected meta_arch={meta_arch}, engine={engine} for regression_length={regression_length}")

# Generate the NMS config JSON that model_optimization_nms.alls references, matching
# the chosen meta_arch's schema. reg_layer/cls_layer are left empty on purpose: DFC
# infers the real layer names from the translated HAR. Thresholds are baked into the
# HEF at compile time and cannot be changed at runtime.
nms_config = {
    "nms_scores_th": 0.25,
    "nms_iou_th": 0.45,
    "image_dims": [img_h, img_w],
    "max_proposals_per_class": 100,
    "classes": num_classes,
    "bbox_decoders": [
        {"name": f"{decoder_prefix}_{s}", "stride": s, "reg_layer": "", "cls_layer": ""}
        for s in (8, 16, 32)
    ],
}
# regression_length only belongs in the yolov8 (DFL) schema; the yolov6 default config
# has no such field. Including it under yolov6 would be meaningless (and reg_len=1 is
# the whole reason we're on yolov6).
if meta_arch == "yolov8":
    nms_config["regression_length"] = regression_length

with open("yolo_nms_config.json", "w") as f:
    json.dump(nms_config, f, indent=2)
print(f"NMS config written to yolo_nms_config.json (meta_arch={meta_arch}, image_dims={[img_h, img_w]})")

# hw_arch is fixed here (not at compile time) and carries through the saved HAR files.
# Options: 'hailo8', 'hailo8l', 'hailo15', 'hailo15l'. Sourced from the profile
# (hailo.hw_arch); defaults to 'hailo8' when no config is passed.
runner = ClientRunner(hw_arch=hw_arch)

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

# model_optimization_nms.alls is a template: it adds nms_postprocess() but leaves
# meta_arch as __META_ARCH__. Render it with the meta_arch selected above (meta_arch
# IS a valid nms_postprocess kwarg, unlike classes/regression_length which live in
# the config JSON). Write the rendered result to a sibling file and load it by path.
with open("model_optimization_nms.alls") as f:
    model_script = f.read()
model_script = model_script.replace("__META_ARCH__", meta_arch).replace("__ENGINE__", engine)
rendered_alls_path = "model_optimization_nms.rendered.alls"
with open(rendered_alls_path, "w") as f:
    f.write(model_script)
runner.load_model_script(rendered_alls_path)

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
