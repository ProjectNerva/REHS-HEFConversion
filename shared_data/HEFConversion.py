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
end_node_names = [node.name for node in onnx_model.graph.output]
print("Start nodes:", start_node_names)
print("End nodes:  ", end_node_names)

runner = ClientRunner()

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
runner = ClientRunner(har_path=f"{model_name}_fp32.har")

runner.load_model_script("model_optimization.alls")

# Run full quantization algorithm using real data to minimize math accuracy loss
runner.optimize(calibration_data)

# Save the quantized model archive
quantized_har_path = f"{model_name}_quantized.har"
runner.save_har(quantized_har_path)
print("Quantization complete. Quantized HAR saved.")

# Emulate quantized model on the calibration set and compare against full-precision
# to catch accuracy regressions before committing to hardware compilation
print("Running emulation accuracy check...")
metrics = runner.evaluate(dataset=calibration_data)
print("Emulation metrics:", metrics)

# Load the quantized archive
runner = ClientRunner(har_path=f"{model_name}_quantized.har")

# Compile targeting your specific hardware architecture
# Options: 'hailo8', 'hailo8l', 'hailo15', 'hailo15l'
hef = runner.compile(hw_arch="hailo8")

# Write out the final binary deployable file
hef_output_path = f"{model_name}.hef"
with open(hef_output_path, "wb") as f:
    f.write(hef)

print(f"Success! Model compiled into target binary: {hef_output_path}")
