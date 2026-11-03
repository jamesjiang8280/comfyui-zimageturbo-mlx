#!/usr/bin/env python3
"""Run Z-Image-Turbo + ControlNet Union on MLX with a *custom* base model.

Why this exists instead of just calling `mflux-generate-z-image-controlnet`:
mflux resolves `--model <path>` through its registry, and a local finetune has no
`controlnet_model` attached, so the stock CLI refuses it ("not a Union ControlNet
model").  We build the ModelConfig ourselves: keep the Union entry's
`controlnet_model`, but point `model_name` at the local directory.  Everything
after that is stock mflux.

Expected layout of `--model DIR` (see ensure_composite() in nodes.py):

    DIR/transformer/  text_encoder/  vae/  tokenizer/     <- base weights
    DIR/controlnet/*.safetensors                           <- Union checkpoint
    DIR/mflux_model.json                                   <- optional (mflux-save)

Usage:
    python zimage_controlnet_run.py --model DIR --control-type canny \\
        --control-image hint.png --prompt "..." --width 768 --height 768 \\
        --steps 8 --seed 42 --output out.png
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="composite dir: base weights + controlnet/")
    ap.add_argument("--control-type", required=True,
                    choices=["canny", "depth", "pose", "hed", "mlsd"])
    ap.add_argument("--control-image", required=True)
    ap.add_argument("--control-strength", type=float, default=0.8,
                    help="per-control strength (model card: 0.65-1.0)")
    ap.add_argument("--controlnet-strength", type=float, default=1.0,
                    help="global multiplier over all controls")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", default="linear")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    control_img = Path(args.control_image).expanduser()
    for p, what in ((model_dir, "model dir"), (control_img, "control image")):
        if not p.exists():
            print(f"[cn] missing {what}: {p}", file=sys.stderr)
            return 2

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
    from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet

    # Keep the Union entry (so controlnet_model is set) but load base weights from DIR.
    # ModelConfig is a plain class, and the registry hands out lru_cached instances --
    # deepcopy so we never mutate the shared entry.
    cfg = copy.deepcopy(ModelConfig.z_image_turbo_controlnet_union_2_1())
    cfg.model_name = str(model_dir)
    print(f"[cn] base={model_dir}  controlnet={cfg.controlnet_model}  "
          f"type={args.control_type} strength={args.control_strength}", flush=True)

    model = ZImageTurboControlnet(model_config=cfg, model_path=str(model_dir))

    spec = ControlSpec(
        type=ControlType(args.control_type),
        image_path=str(control_img),
        strength=float(args.control_strength),
    )
    print("[cn] generating ...", flush=True)
    image = model.generate_image(
        seed=args.seed,
        prompt=args.prompt,
        controls=[spec],
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        controlnet_strength=args.controlnet_strength,
        scheduler=args.scheduler,
    )
    image.save(path=args.output)
    print(f"[cn] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
