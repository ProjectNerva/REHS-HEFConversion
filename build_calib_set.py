import glob
import sys

import numpy as np
from PIL import Image

COLOR = (114, 114, 114)

def letterbox(img, new_h, new_w):
    w, h = img.size
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), COLOR)
    canvas.paste(resized, (0, 0))
    return canvas


def main(input_dir, output_path, new_h, new_w):
    paths = sorted(glob.glob(f"{input_dir}/*.JPG") + glob.glob(f"{input_dir}/*.jpg"))
    print(f"Found {len(paths)} images")

    arr = np.empty((len(paths), new_h, new_w, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        arr[i] = np.array(letterbox(img, new_h, new_w), dtype=np.float32)

    # Keep NHWC: Hailo's ClientRunner.optimize() calibration dataset expects
    # (N, H, W, C) regardless of the ONNX model's own NCHW input layout.
    print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())
    np.save(output_path, arr)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python build_calib_set.py <input_dir> <output_npy_path> <height> <width>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
