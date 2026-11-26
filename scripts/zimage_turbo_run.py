#!/usr/bin/env python3
"""Run Z-Image-Turbo on MLX with mflux's StepCache (TeaCache-style) enabled.

mflux ships `mflux.models.common.step_cache.StepCache`, but the Z-Image variant
never wires it into its denoise loop and the CLI exposes no flag for it.  This
driver runs the same stock model while injecting the cache from the outside:

  * `Config.scheduler` is wrapped so every scheduler step records which step is
    next (`t+1`).  The loop is strictly predict -> step -> eval, so this counter
    is an exact step index at predict time.
  * `ZImage._predict` is wrapped so a step whose index is in the cache's skip
    set returns the previous step's prediction instead of running the DiT.

Everything else (latents, scheduler, VAE decode, callbacks) stays stock mflux.

Usage:
    python zimage_turbo_run.py --model DIR --prompt "..." --steps 8 --seed 42 \\
        --step-cache-ratio 0.3 --output out.png
    python zimage_turbo_run.py --model DIR --prompt "..." --image photo.png \\
        --image-strength 0.6 --output out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STATE: dict = {"cfg": None, "cache": None, "next_t": 0}


def _install_patches(ratio: float, protect: int) -> None:
    """Wire StepCache into the stock loop (see module docstring)."""
    from mflux.models.common.config.config import Config
    from mflux.models.common.step_cache.step_cache import StepCache
    from mflux.models.z_image.variants.z_image import ZImage

    # mflux disables the cache for runs shorter than MIN_STEPS (10) because skipping
    # does change the image. ComfyUI users opt in explicitly and Z-Image-Turbo lives
    # at 8-9 steps, so lower the floor to 2 (same trade-off the Qwen node makes).
    StepCache.MIN_STEPS = 2

    orig_scheduler_prop = Config.scheduler.fget
    orig_predict = ZImage._predict

    def scheduler(self):
        sched = orig_scheduler_prop(self)
        if not getattr(sched, "_zimage_step_cache_patched", False):
            orig_step = sched.step

            def step(noise, timestep, latents, **kw):
                out = orig_step(noise=noise, timestep=timestep, latents=latents, **kw)
                STATE["next_t"] = int(timestep) + 1
                return out

            sched.step = step
            sched._zimage_step_cache_patched = True      # type: ignore[attr-defined]
            STATE["cfg"] = self
            STATE["next_t"] = self.init_time_step
            cache = StepCache.for_run(self, ratio)
            if protect > 0 and cache.active:
                # mflux only protects a fraction of both ends; for short Turbo runs
                # the first steps carry the composition, so let the caller fence
                # them off explicitly.
                floor = self.init_time_step + protect
                kept = frozenset(t for t in cache.skip_steps if t >= floor)
                if kept != cache.skip_steps:
                    cache = StepCache(kept)
            STATE["cache"] = cache
        return sched

    def predict(transformer):
        inner = orig_predict(transformer)

        def cached(**kwargs):
            cache = STATE.get("cache")
            if cache is None or not cache.active:
                return inner(**kwargs)
            t = STATE["next_t"]
            hit = cache.reuse(t)
            if hit is not None:
                return hit
            out = inner(**kwargs)
            cache.store(out)
            return out

        return cached

    Config.scheduler = property(scheduler)            # type: ignore[assignment]
    ZImage._predict = staticmethod(predict)           # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local diffusers dir or mflux-saved dir")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", default="linear")
    ap.add_argument("--quantize", type=int, default=None, choices=[3, 4, 5, 6, 8])
    ap.add_argument("--image", default=None, help="init image for image-to-image")
    ap.add_argument("--image-strength", type=float, default=0.6,
                    help="larger keeps more of --image (mflux semantics)")
    ap.add_argument("--step-cache-ratio", type=float, default=0.0,
                    help="fraction of denoise steps to reuse (0 = off, 0.2-0.4 typical)")
    ap.add_argument("--step-cache-protect", type=int, default=1,
                    help="never skip the first N steps (composition is decided there)")
    ap.add_argument("--lora", action="append", nargs="+", default=None,
                    metavar=("PATH", "SCALE"), help="repeatable: --lora path 0.7")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    lora_paths: list[str] = []
    lora_scales: list[float] = []
    for group in args.lora or []:
        lora_paths.append(group[0])
        lora_scales.append(float(group[1]) if len(group) > 1 else 1.0)
    if lora_paths:
        print(f"[run] loras: {list(zip(lora_paths, lora_scales))}", flush=True)

    model_dir = Path(args.model).expanduser()
    if not model_dir.is_dir():
        print(f"[run] missing model dir: {model_dir}", file=sys.stderr)
        return 2

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage

    ratio = max(0.0, min(0.95, float(args.step_cache_ratio)))
    if ratio > 0:
        _install_patches(ratio, max(0, int(args.step_cache_protect)))

    cfg = ModelConfig.z_image_turbo()
    try:
        cfg.model_name = str(model_dir)
    except Exception:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.model_name = str(model_dir)

    print(f"[run] model={model_dir.name} steps={args.steps} size={args.width}x{args.height} "
          f"quantize={args.quantize} step_cache={ratio or 'off'}", flush=True)

    model = ZImage(
        model_config=cfg,
        model_path=str(model_dir),
        quantize=args.quantize,
        lora_paths=lora_paths or None,
        lora_scales=lora_scales or None,
    )

    image = model.generate_image(
        seed=args.seed,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        scheduler=args.scheduler,
        image_path=args.image,
        image_strength=args.image_strength if args.image else None,
    )
    image.save(path=args.output)

    cache = STATE.get("cache")
    if cache is not None and cache.active:
        total = args.steps - STATE["cfg"].init_time_step
        print(f"[run] step-cache: 跳过 {len(cache.skip_steps)}/{total} 步 -> {sorted(cache.skip_steps)}",
              flush=True)
    print(f"[run] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
