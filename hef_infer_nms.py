import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw

PAD_COLOR = (114, 114, 114)


def letterbox(img, new_h, new_w):
    """Resize keeping aspect ratio and pad to (new_h, new_w). Returns (canvas, ratio)."""
    w, h = img.size
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), PAD_COLOR)
    canvas.paste(resized, (0, 0))
    return canvas, r


class HailoModel:
    def __init__(self, hef_path):
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self._hef = HEF(hef_path)
        self._device = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._device.configure(self._hef, configure_params)[0]
        self._ng_params = self._network_group.create_params()

        in_info = self._hef.get_input_vstream_infos()[0]
        self._input_name = in_info.name
        self.input_h, self.input_w = in_info.shape[0], in_info.shape[1]

        self._in_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        self._out_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        self._InferVStreams = __import__(
            "hailo_platform", fromlist=["InferVStreams"]
        ).InferVStreams

    def infer(self, image_nhwc_uint8):
        """Run inference. Returns (N, 6) float32 array: [y_min, x_min, y_max, x_max, score, class_id].
        Only real detections are included — no padding rows.
        """
        with self._network_group.activate(self._ng_params):
            with self._InferVStreams(
                self._network_group, self._in_params, self._out_params
            ) as pipeline:
                results = pipeline.infer({self._input_name: image_nhwc_uint8})

        raw = next(iter(results.values()))

        # HailoRT NMS output: list[batch] → list[class] → ndarray(N_cls, 5)
        # Each detection row: [y_min, x_min, y_max, x_max, score]
        if isinstance(raw, list):
            per_class = raw[0]  # first (only) batch item
            rows = []
            for cls_id, cls_dets in enumerate(per_class):
                if cls_dets is None or len(cls_dets) == 0:
                    continue
                cls_dets = np.asarray(cls_dets, dtype=np.float32)
                if cls_dets.ndim == 1:
                    cls_dets = cls_dets[None, :]
                class_col = np.full((len(cls_dets), 1), cls_id, dtype=np.float32)
                rows.append(np.concatenate([cls_dets, class_col], axis=1))
            return np.concatenate(rows, axis=0) if rows else np.zeros((0, 6), dtype=np.float32)

        # Fallback: flat tensor output (batch, max_det, 6)
        raw = np.asarray(raw)
        return raw[0] if raw.ndim == 3 else raw


def load_labels(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Hailo NMS-baked HEF inference (no host-side decode or NMS)"
    )
    ap.add_argument("hef", help="Path to the NMS-compiled .hef (built with HEFConversion-pB.py)")
    ap.add_argument("image", help="Path to an input image")
    ap.add_argument("--labels", default=None, help="Optional class-name file (one per line)")
    ap.add_argument("--out", default=None, help="Optional path to save an annotated image")
    args = ap.parse_args()

    labels = load_labels(args.labels)

    try:
        model = HailoModel(args.hef)
    except ImportError:
        print(
            "ERROR: hailo_platform (HailoRT) is not installed. Run this on the Pi/AI HAT "
            "with HailoRT installed, not the build container.",
            file=sys.stderr,
        )
        sys.exit(1)

    orig = Image.open(args.image).convert("RGB")
    canvas, ratio = letterbox(orig, model.input_h, model.input_w)
    inp = np.ascontiguousarray(np.array(canvas, dtype=np.uint8)[None, ...])  # (1, H, W, 3)

    detections = model.infer(inp)  # (max_detections, 6)

    # Filter padding rows (HailoRT pads unused slots with zeros)
    valid = detections[detections[:, 4] > 0]

    ow, oh = orig.size
    print(f"{len(valid)} detections:")
    draw = ImageDraw.Draw(orig) if args.out else None

    for det in valid:
        y1, x1, y2, x2, score, cls_id = det
        # Coords are normalized [0,1] → scale to network input pixels → un-letterbox
        x1, y1, x2, y2 = (x1 * model.input_w) / ratio, (y1 * model.input_h) / ratio, (x2 * model.input_w) / ratio, (y2 * model.input_h) / ratio
        x1, x2 = max(0.0, min(float(x1), ow)), max(0.0, min(float(x2), ow))
        y1, y2 = max(0.0, min(float(y1), oh)), max(0.0, min(float(y2), oh))
        cls_id = int(cls_id)
        name = labels[cls_id] if labels and cls_id < len(labels) else str(cls_id)
        print(f"  {name:>15}  {score:.3f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        if draw:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1, max(0, y1 - 10)), f"{name} {score:.2f}", fill=(255, 0, 0))

    if args.out:
        orig.save(args.out)
        print(f"Annotated image saved to {args.out}")


if __name__ == "__main__":
    main()
