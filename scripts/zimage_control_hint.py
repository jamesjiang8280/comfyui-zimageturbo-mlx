#!/usr/bin/env python3
"""Render the ControlNet hint mflux would compute, so the node can show it.

mflux's `ZImageControlnetUtil._preprocess(img, control_type)` is what turns a
photo into the actual canny/depth/pose/hed/mlsd signal.  The sampler used to run
it invisibly inside the generation subprocess; the node now runs the same code
here, once, and hands the resulting image back to the graph.

Usage:
    python zimage_control_hint.py --image in.png --type pose --output hint.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--type", required=True,
                    choices=["canny", "depth", "pose", "hed", "mlsd"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = Path(args.image).expanduser()
    if not src.is_file():
        print(f"[hint] missing input image: {src}", file=sys.stderr)
        return 2

    from PIL import Image

    from mflux.models.z_image.variants.controlnet.control_types import ControlType
    from mflux.models.z_image.variants.controlnet.controlnet_util import ZImageControlnetUtil

    img = Image.open(src).convert("RGB")
    hint = ZImageControlnetUtil._preprocess(img, ControlType(args.type))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    hint.save(out)
    print(f"[hint] {args.type}: {img.size} -> {hint.size} "
          f"mode={hint.mode} bytes={out.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
