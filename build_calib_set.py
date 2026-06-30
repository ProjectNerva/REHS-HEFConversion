import argparse
import glob
import os

import numpy as np
from PIL import Image


def stretch(img, height, width):
    return img.resize((width, height), Image.BILINEAR)


def main(input_dir, height, width):
    paths = sorted(glob.glob(f"{input_dir}/*.JPG") + glob.glob(f"{input_dir}/*.jpg"))
    print(f"Found {len(paths)} images")

    arr = np.empty((len(paths), height, width, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        arr[i] = np.array(stretch(img, height, width), dtype=np.float32)

    print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())

    dir_name = os.path.basename(os.path.normpath(input_dir))
    output_path = f"calib_{dir_name}_{height}x{width}.npy"
    np.save(output_path, arr)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("height", type=int, help="stretch target height, i.e. NCHW dim 2")
    parser.add_argument("width", type=int, help="stretch target width, i.e. NCHW dim 3")
    args = parser.parse_args()
    main(args.input_dir, args.height, args.width)
