import glob
import sys

import numpy as np
from PIL import Image

NEW_H, NEW_W = 608, 416
COLOR = (114, 114, 114)


def letterbox(img):
    w, h = img.size
    r = min(NEW_W / w, NEW_H / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (NEW_W, NEW_H), COLOR)
    left = (NEW_W - nw) // 2
    top = (NEW_H - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def main(input_dir, output_path):
    paths = sorted(glob.glob(f"{input_dir}/*.JPG") + glob.glob(f"{input_dir}/*.jpg"))
    print(f"Found {len(paths)} images")

    arr = np.empty((len(paths), NEW_H, NEW_W, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        arr[i] = np.array(letterbox(img), dtype=np.float32)

    print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())
    np.save(output_path, arr)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
