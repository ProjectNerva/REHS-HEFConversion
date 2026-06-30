import sys

import onnx
import onnxruntime as ort
from onnx import utils

onnx_path = sys.argv[1]
output_path = sys.argv[2]
height = int(sys.argv[3])  # NCHW dim 2 -- same height-then-width order as the rest of this repo
width = int(sys.argv[4])   # NCHW dim 3


def find_detection_nodes(onnx_path):
    """
    Walk backwards from the model output to find the detection head cut points.
    Returns (end_tensor_names, decoded_output) where end_tensor_names are the output
    tensor names of the final Conv layers (cv2/cv3 branches at each scale) and
    decoded_output is the graph's final detection tensor.
    Works for any YOLO11/YOLO8 architecture without hardcoded paths.
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    tensor_to_node = {out: node for node in graph.node for out in node.output}
    decoded_output = graph.output[0].name

    tensor_names = []
    visited = set()

    def trace(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        if tensor not in tensor_to_node:
            return
        node = tensor_to_node[tensor]
        if node.op_type == 'Conv':
            if node.output[0] not in tensor_names:
                tensor_names.append(node.output[0])
            return
        for inp in node.input:
            trace(inp)

    trace(decoded_output)
    if not tensor_names:
        raise RuntimeError("Could not detect detection head Conv nodes — check ONNX graph structure.")
    return tensor_names, decoded_output


# 1. fix the dynamic input shape so the shape-dependent anchor-grid ops can fold to constants
model = onnx.load(onnx_path)
for inp in model.graph.input:
    if inp.name == 'images':
        for d, v in zip(inp.type.tensor_type.shape.dim, [1, 3, height, width]):
            d.ClearField('dim_param')
            d.dim_value = v
onnx.save(model, 'fixed_shape.onnx')

# 2. constant-fold the now-static anchor-grid computation (no hardware-specific layout transforms)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.optimized_model_filepath = 'optimized_shape.onnx'
ort.InferenceSession('fixed_shape.onnx', sess_options=so, providers=['CPUExecutionProvider'])

# 3. auto-detect subgraph boundaries from the optimized graph
end_node_outputs, decoded_output = find_detection_nodes('optimized_shape.onnx')
print(f"Detected {len(end_node_outputs)} end nodes: {end_node_outputs}")
print(f"Decoded output: {decoded_output}")

# 4. extract just the decode subgraph: raw conv outputs -> decoded (boxes, class_probs) tensor
utils.extract_model('optimized_shape.onnx', output_path, end_node_outputs, [decoded_output])
print(f"Saved {output_path}")
