# REHS-HEFConversion

Convert an Ultralytics YOLO model (.pt) into a Hailo `.hef` file for deployment on Hailo-8 accelerators. The pipeline handles ONNX export, graph simplification, calibration dataset preparation, quantization, and hardware compilation.

## Prerequisites
- Docker needs to be downloaded + an account is needed. On macOS, enable **"Use Rosetta for x86_64/amd64 emulation on Apple Silicon"** in Docker Desktop settings.
- The Hailo Dataflow Compiler wheel is required. This repo targets Python 3.10. Download from the Hailo Developer Zone: select **AI Software Suite → Dataflow Compiler → x86 → Linux → 3.10**, filter by **Archive**, and download version **3.33.1**. Place the `.whl` file in the root of this repo.
- The `.pt` model file you want to convert.
- A folder of representative calibration images (100–1000 real samples from your dataset).

## General Workflow

### 1. Build the Docker image
```
docker build -t hef-conversion .
```

### 2. Export and simplify the ONNX model
Run locally. This converts the YOLO `.pt` to ONNX, simplifies the graph, and lowers the IR version if needed.
```
python main.py <model.pt>
```
A simplified `.onnx` file will be saved in the same directory. Move it into `shared_data/`.

### 3. Build the calibration dataset
Run locally. Letterboxes images to match the model's input size and saves them as a NumPy array. Height and width must match the export size used in step 2 (default: 608 × 416).
```
python build_calib_set.py <input_image_dir> <output.npy> <height> <width>
```
Move the output `.npy` file into `shared_data/`.

### 4. Run the Docker container
```
docker run -it -v "$(pwd)/shared_data:/app/shared_data" hef-conversion
```

If you want to restart an older container you left
```
docker start -ai <container_id>
```

### 5. Run the HEF conversion inside the container
```
cd shared_data
python3 HEFConversion.py <model_name> <model.onnx> <calibration.npy>
```
This will:
- Parse the ONNX graph into a Hailo Archive (HAR)
- Apply the model optimization script (`model_optimization.alls`) for on-chip normalization and max compiler optimization
- Quantize using your real calibration data
- Run emulation metrics to verify accuracy before compilation
- Compile to the final `.hef` binary

The output `<model_name>.hef` file is ready for deployment on Hailo-8 hardware.

### 6. Running Inference using the HEF File

Run this on the target device (e.g. Raspberry Pi with Hailo AI HAT) with HailoRT installed — **not** inside the build container.

```
python hef_infer.py <model.hef> <image>
```

Post-processing (YOLO decode + NMS) runs on the host CPU. The script auto-detects the detection head layout from the output tensor shapes, so no extra configuration is needed for standard YOLO models.

Optional arguments:
| Argument | Default | Description |
|---|---|---|
| `--score-thr` | `0.25` | Confidence threshold |
| `--iou-thr` | `0.45` | NMS IoU threshold |
| `--max-det` | `300` | Max detections after NMS |
| `--labels` | — | Path to a class names file (one name per line) |
| `--out` | — | Path to save an annotated output image |
| `--reg-max` | auto | Override DFL reg_max (default: 16 for YOLO11/v8, 1 for DFL-free models) |
| `--num-classes` | auto | Override class count if auto-detection is ambiguous |

Example with all options:
```
python hef_infer.py model.hef photo.jpg --score-thr 0.3 --labels classes.txt --out result.jpg
```

Detections are printed to stdout in the format:
```
3 detections:
          person  0.872  [120, 45, 380, 510]
             dog  0.761  [200, 300, 450, 600]
             cat  0.643  [10, 20, 150, 200]
```
