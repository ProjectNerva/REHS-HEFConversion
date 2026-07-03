import sys
import numpy as np
import onnx
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

runner.load_model_script("model_optimization.alls")

# lazy loading gen to prevent 16 GB crash
# feeds one by one instead of all at once
def calibration_feed():
    raw_data = np.load(calibration_data)
    for sample in raw_data:
        yield {start_node_names[0]: np.expand_dims(sample, axis = 0) if sample.ndim == 3 else sample}

# Run full quantization algorithm using real data to minimize math accuracy loss
runner.optimize(calibration_feed())

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
hef_output_path = f"{model_name}.hef"
with open(hef_output_path, "wb") as f:
    f.write(hef)

print(f"Success! Model compiled into target binary: {hef_output_path}")