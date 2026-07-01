import argparse
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

PAD_COLOR = (114, 114, 114)
DEFAULT_REG_MAX = 16  # Ultralytics DFL convention: box branch has 4 * reg_max channels


# --------------------------------------------------------------------------- #
# Preprocessing (mirrors build_calib_set.py: aspect-preserving, pad 114, top-left)
# --------------------------------------------------------------------------- #
def letterbox(img, new_h, new_w):
    """Resize keeping aspect ratio and pad to (new_h, new_w). Returns (canvas, ratio)."""
    w, h = img.size
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), PAD_COLOR)
    canvas.paste(resized, (0, 0))
    return canvas, r


# --------------------------------------------------------------------------- #
# Pure-numpy decode + NMS (no hardware needed; unit-testable off the Pi)
# --------------------------------------------------------------------------- #
def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dfl_decode(reg, reg_max):
    """reg: (H, W, 4*reg_max) distribution logits -> (H, W, 4) l,t,r,b distances (stride units)."""
    h, w, _ = reg.shape
    reg = reg.reshape(h, w, 4, reg_max)
    prob = _softmax(reg, axis=-1)
    bins = np.arange(reg_max, dtype=np.float32)
    return np.sum(prob * bins, axis=-1)


def decode_scale(reg, cls, stride, reg_max):
    """Decode one scale's raw (reg, cls) maps into pixel-space boxes + class scores."""
    h, w, _ = cls.shape
    if reg_max > 1:
        dist = dfl_decode(reg, reg_max)  # DFL head: (H, W, 4*reg_max) -> (H, W, 4)
    else:
        dist = reg.reshape(h, w, 4)  # DFL-free head (e.g. YOLO26): 4 channels are l,t,r,b directly
    gx = np.arange(w, dtype=np.float32) + 0.5
    gy = np.arange(h, dtype=np.float32) + 0.5
    gxv, gyv = np.meshgrid(gx, gy)  # (H, W)
    left, top, right, bot = dist[..., 0], dist[..., 1], dist[..., 2], dist[..., 3]
    x1 = (gxv - left) * stride
    y1 = (gyv - top) * stride
    x2 = (gxv + right) * stride
    y2 = (gyv + bot) * stride
    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
    scores = _sigmoid(cls).reshape(-1, cls.shape[-1])
    return boxes, scores


def _greedy_nms(boxes, scores, iou_thr):
    """Standard greedy NMS on a single class. Returns kept indices into boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode_and_nms(scales, score_thr, iou_thr, max_det):
    """
    scales: list of (reg_map, cls_map, stride, reg_max).
    Returns list of (x1, y1, x2, y2, score, class_id) in network-input pixel coords.
    """
    all_boxes, all_scores = [], []
    for reg, cls, stride, reg_max in scales:
        boxes, scores = decode_scale(reg, cls, stride, reg_max)
        all_boxes.append(boxes)
        all_scores.append(scores)
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    # One label per box (argmax) — standard YOLO inference behaviour.
    cls_ids = np.argmax(scores, axis=1)
    conf = scores[np.arange(len(scores)), cls_ids]
    mask = conf >= score_thr
    boxes, conf, cls_ids = boxes[mask], conf[mask], cls_ids[mask]

    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        for k in _greedy_nms(boxes[idx], conf[idx], iou_thr):
            b = boxes[idx][k]
            dets.append((b[0], b[1], b[2], b[3], float(conf[idx][k]), int(c)))
    dets.sort(key=lambda d: -d[4])
    return dets[:max_det]


# --------------------------------------------------------------------------- #
# Auto-detect the detection-head layout from the raw output tensor shapes
# --------------------------------------------------------------------------- #
def build_scales(outputs, input_h, input_w, reg_max=None, num_classes=None):
    """
    outputs: dict {name: np.ndarray of shape (H, W, C)} (batch already removed).
    Groups the 2*N raw maps into N scales, identifying which map is the DFL box
    (reg) branch vs the class branch, and infers the stride per scale.

    Returns list of (reg_map, cls_map, stride, reg_max) sorted by stride ascending.
    """
    channels = sorted({arr.shape[-1] for arr in outputs.values()})

    # Determine which channel count is the box (reg) branch and which is cls.
    # DFL heads expose 4*reg_max box channels (reg_max=16 -> 64); DFL-free heads
    # (e.g. YOLO26) expose 4 box channels directly (reg_max=1). The cls branch has
    # num_classes channels.
    if reg_max is not None:
        reg_ch = 4 * reg_max
    elif num_classes is not None:
        others = [c for c in channels if c != num_classes]
        if len(others) != 1:
            raise ValueError(
                f"With --num-classes {num_classes}, expected exactly one other channel count "
                f"for the box branch, but channels are {channels}."
            )
        reg_ch = others[0]
    elif 4 * DEFAULT_REG_MAX in channels:
        reg_ch = 4 * DEFAULT_REG_MAX  # DFL head, reg_max=16 -> 64
    elif 4 in channels:
        reg_ch = 4  # DFL-free head, box branch is 4 channels
    else:
        raise ValueError(
            f"Could not identify the box branch. Output channel counts are {channels}; "
            f"expected 64 (DFL reg_max=16) or 4 (DFL-free). Pass --reg-max or --num-classes "
            "to disambiguate."
        )

    if reg_ch not in channels:
        raise ValueError(
            f"Box branch channel count {reg_ch} not present in outputs {channels}. "
            "Pass --reg-max or --num-classes to disambiguate."
        )
    cls_ch_candidates = [c for c in channels if c != reg_ch]
    if len(cls_ch_candidates) != 1:
        raise ValueError(
            f"Could not identify a single class branch. Channel counts {channels} left "
            f"class candidates {cls_ch_candidates}. Pass --num-classes to disambiguate."
        )
    reg_max = reg_ch // 4

    # Pair up reg/cls maps that share the same spatial resolution (one scale each).
    by_res = defaultdict(dict)
    for name, arr in outputs.items():
        h, w, c = arr.shape
        role = "reg" if c == reg_ch else "cls"
        if role in by_res[(h, w)]:
            raise ValueError(f"Two '{role}' maps at resolution {(h, w)}; cannot pair scales.")
        by_res[(h, w)][role] = arr

    scales = []
    for (h, w), pair in by_res.items():
        if "reg" not in pair or "cls" not in pair:
            raise ValueError(f"Resolution {(h, w)} is missing a reg or cls map: {list(pair)}")
        # Stride from input/grid ratio (rounded; matches 8/16/32 for standard YOLO heads).
        stride = int(round(input_h / h))
        if int(round(input_w / w)) != stride:
            raise ValueError(f"Non-square stride at {(h, w)}: {input_h}/{h} vs {input_w}/{w}")
        scales.append((pair["reg"], pair["cls"], stride, reg_max))

    scales.sort(key=lambda s: s[2])  # ascending stride (finest grid first)
    return scales


# --------------------------------------------------------------------------- #
# HailoRT inference wrapper (imported lazily so this file works off-hardware)
# --------------------------------------------------------------------------- #
class HailoModel:
    def __init__(self, hef_path):
        from hailo_platform import (  # noqa: import here so the module loads without HailoRT
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
        """image_nhwc_uint8: (1, H, W, 3) uint8. Returns {name: (H, W, C)} float32."""
        with self._network_group.activate(self._ng_params):
            with self._InferVStreams(
                self._network_group, self._in_params, self._out_params
            ) as pipeline:
                results = pipeline.infer({self._input_name: image_nhwc_uint8})
        return {name: arr[0] for name, arr in results.items()}  # drop batch dim


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_labels(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Hailo HEF inference + host-side YOLO decode/NMS")
    ap.add_argument("hef", help="Path to the compiled .hef")
    ap.add_argument("image", help="Path to an input image")
    ap.add_argument("--score-thr", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou-thr", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections after NMS")
    ap.add_argument("--reg-max", type=int, default=None, help="Override DFL reg_max (default: auto/16)")
    ap.add_argument("--num-classes", type=int, default=None, help="Override class count (default: auto)")
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
    # np.array (not asarray) forces a writeable, contiguous copy; HailoRT's infer()
    # rejects the read-only buffer that PIL images expose.
    inp = np.ascontiguousarray(np.array(canvas, dtype=np.uint8)[None, ...])  # (1, H, W, 3)

    outputs = model.infer(inp)
    scales = build_scales(
        outputs, model.input_h, model.input_w, reg_max=args.reg_max, num_classes=args.num_classes
    )
    dets = decode_and_nms(scales, args.score_thr, args.iou_thr, args.max_det)

    # Un-letterbox: content was pasted top-left with scale `ratio`, so divide back out.
    ow, oh = orig.size
    print(f"{len(dets)} detections:")
    draw = ImageDraw.Draw(orig) if args.out else None
    for x1, y1, x2, y2, score, cls_id in dets:
        x1, y1, x2, y2 = x1 / ratio, y1 / ratio, x2 / ratio, y2 / ratio
        x1, x2 = max(0.0, min(x1, ow)), max(0.0, min(x2, ow))
        y1, y2 = max(0.0, min(y1, oh)), max(0.0, min(y2, oh))
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