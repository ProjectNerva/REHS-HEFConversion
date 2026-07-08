import glob
import random
import sys

import numpy as np
from PIL import Image

COLOR = (114, 114, 114)
DEFAULT_NUM_IMAGES = 1024  # matches calibset_size in model_optimization*.alls
SEED = 0  # fixed seed so the same input_dir reproducibly picks the same subset

def letterbox(img, new_h, new_w):
    w, h = img.size
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), COLOR)
    canvas.paste(resized, (0, 0))
    return canvas


def main(input_dir, output_path, new_h, new_w, num_images=DEFAULT_NUM_IMAGES):
    paths = sorted(glob.glob(f"{input_dir}/*.JPG") + glob.glob(f"{input_dir}/*.jpg"))
    print(f"Found {len(paths)} images")

    if num_images < len(paths):
        paths = random.Random(SEED).sample(paths, num_images)
    elif num_images > len(paths):
        print(f"WARNING: requested {num_images} images but only {len(paths)} available — using all of them")
    print(f"Selected {len(paths)} images for calibration")

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
        print("Usage: python build_calib_set.py <input_dir> <output_npy_path> <height> <width> [num_images]")
        print(f"  num_images defaults to {DEFAULT_NUM_IMAGES}; randomly sampled if the dir has more.")
        sys.exit(1)
    num_images = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_NUM_IMAGES
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), num_images)
