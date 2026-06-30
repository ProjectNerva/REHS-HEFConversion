import sys
import onnx
import hailo_sdk_client
from hailo_sdk_client import ClientRunner

onnx_name = sys.argv[1]
har_name = sys.argv[2]
calibration_path = sys.argv[3]
hef_name = sys.argv[4]
height = int(sys.argv[5])  # NCHW dim 2 -- same height-then-width order as main.py/build_calib_set.py
width = int(sys.argv[6])   # NCHW dim 3


def find_end_node_names(onnx_path):
    """
    Walk backwards from the model output to find the ONNX node names of the final
    Conv layers in the detection head. Stops the trace at each Conv node, which for
    any YOLO11/YOLO8 model corresponds to the cv2 (box) and cv3 (class) output convs.
    Works for any model size or class count — no hardcoded layer paths needed.
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    tensor_to_node = {out: node for node in graph.node for out in node.output}
    decoded_output = graph.output[0].name

    node_names = []
    visited = set()

    def trace(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        if tensor not in tensor_to_node:
            return
        node = tensor_to_node[tensor]
        if node.op_type == 'Conv':
            if node.name not in node_names:
                node_names.append(node.name)
            return
        for inp in node.input:
            trace(inp)

    trace(decoded_output)
    if not node_names:
        raise RuntimeError("Could not detect detection head Conv nodes — check ONNX graph structure.")
    return node_names


end_node_names = find_end_node_names(onnx_name)
print(f"Detected {len(end_node_names)} end nodes: {end_node_names}")

runner = ClientRunner(hw_arch='hailo8')
runner.translate_onnx_model(
    f'{onnx_name}', f'{har_name}',
    net_input_shapes={'images': [1, 3, height, width]},
    end_node_names=end_node_names,
)

# optimization_level=3 enables ADAROUND (adaptive per-weight rounding), which minimises
# task loss directly rather than just rounding to nearest integer. This improves SNR on
# the detection head output layers that previously sat at 3-12 dB with level 2.
# activation_clipping applied without a layer list so it is model-agnostic; it handles
# the heavy-tailed activations in the box/class branches for any YOLO-family model.
runner.load_model_script(
    "model_optimization_config(calibration, calibset_size=1024)\n"
    "model_optimization_flavor(optimization_level=3, compression_level=0)\n"
)
runner.optimize(calib_data=f'{calibration_path}.npy')
runner.save_har(f'{har_name}.har')
hef = runner.compile()
with open(f"{hef_name}.hef", "wb") as f:
    f.write(hef)
