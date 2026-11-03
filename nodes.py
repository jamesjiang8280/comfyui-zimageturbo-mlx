"""Z-Image-Turbo on Apple Silicon via MLX -- ComfyUI nodes.

    ComfyUI (Python)  --subprocess-->  mflux z-image-turbo CLI (MLX)
                                            |
                              Tongyi-MAI/Z-Image-Turbo (Apache-2.0)

Why this exists: ComfyUI's built-in Z-Image nodes run on PyTorch/MPS.  On Apple
Silicon the same weights run several times faster through MLX, which is what
mflux provides.  These nodes keep the ComfyUI graph shape (prompt + size + seed
-> IMAGE) and delegate the diffusion to mflux.

Nodes
-----
* ``ZImage MLX Loader``        -> ZIMAGE_MLX_CONFIG
* ``ZImage MLX Text to Image`` -> IMAGE

Turbo facts worth knowing before you wire it up
-----------------------------------------------
* It is guidance-distilled: mflux forces guidance to 0 and never encodes a
  negative prompt, so this node exposes neither.
* 8-9 steps is the design point (mflux defaults to 9).  More steps mostly cost
  time.
* Low seed variance is a documented model trait, not a node bug.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .runner import ZImageRunError, find_python, run_controlnet, run_generate

try:  # optional: progress bar + interruption, only available inside ComfyUI
    import comfy.model_management as comfy_mm
    import comfy.utils as comfy_utils
except Exception:  # pragma: no cover
    comfy_mm = None
    comfy_utils = None


VERSION = "1.0.0"

DEFAULT_REPO = "Tongyi-MAI/Z-Image-Turbo"
QUANTS = ["4", "8", "6", "bf16"]

_FIXES: list[str] = []


# ---------------------------------------------------------------- helpers ---


def _fix(name: str, msg: str) -> None:
    _FIXES.append(f"{name}: {msg}")


def _report_fixes() -> None:
    if _FIXES:
        print("[ZImage-MLX] 参数已自动规整：" + "; ".join(_FIXES), flush=True)
        _FIXES.clear()


def _as_int(value: Any, default: int, lo: int, hi: int, name: str) -> int:
    try:
        iv = int(float(value))
    except (TypeError, ValueError):
        _fix(name, f"invalid {value!r} -> {default}")
        return default
    if iv < lo or iv > hi:
        clamped = max(lo, min(hi, iv))
        _fix(name, f"{iv} 超出 [{lo},{hi}] -> {clamped}")
        iv = clamped
    return iv


def _as_str(value: Any, default: str, name: str, allow_numbers: bool = False) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, bool):
        _fix(name, f"bool {value!r} -> default")
        return default
    if allow_numbers and isinstance(value, (int, float)):
        return str(value)
    _fix(name, f"str expected, got {value!r} -> default")
    return default


def _snap32(value: int, name: str) -> int:
    """Z-Image VAE is 8x and the DiT patch is 2x2 -> multiples of 32 are safe."""
    snapped = max(256, min(1536, int(round(value / 32.0)) * 32))
    if snapped != value:
        _fix(name, f"{value} 不是 32 的倍数 -> {snapped}")
    return snapped


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def mlx_root() -> Path:
    env = os.environ.get("ZIMAGE_MLX_MODELS_DIR", "").strip()
    if env:
        return Path(os.path.expanduser(env))
    return Path.home() / "ComfyUI-Shared" / "models" / "MLX"


def default_model_dir() -> Path:
    return mlx_root() / "Z-Image-Turbo"


def _is_zimage_diffusers(path: Path) -> bool:
    """A diffusers snapshot of the Z-Image DiT (not just any transformer dir)."""
    cfg = path / "transformer" / "config.json"
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") == "ZImageTransformer2DModel"
    except Exception:
        return False


def _is_mflux_saved(path: Path) -> bool:
    """An mflux-save output (quantized on disk); identified by its model json."""
    f = path / "mflux_model.json"
    if not f.is_file():
        return False
    try:
        blob = f.read_text(encoding="utf-8").lower()
    except Exception:
        return False
    return "z-image" in blob or "zimage" in blob


def _single_file_is_zimage(path: Path) -> bool:
    """Peek at the safetensors header: ComfyUI Z-Image DiTs carry these keys."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(min(n, 4 << 20)))
    except Exception:
        return False
    keys = [k for k in header if k != "__metadata__"]
    has_prefix = any(k.startswith("model.diffusion_model.") for k in keys)
    has_marker = any(k.endswith(("x_embedder.weight", "cap_embedder.1.weight")) for k in keys)
    return has_prefix and has_marker


def _saved_source(path: Path) -> str:
    """Auto-prepared dirs record where they came from, so we never convert twice."""
    f = path / ".source"
    try:
        return f.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def model_options() -> list[str]:
    """Dropdown contents: ready MLX models first, then single-file checkpoints."""
    opts: list[str] = []
    root = mlx_root()
    if root.is_dir():
        for d in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if not d.is_dir():
                continue
            if _is_mflux_saved(d) or _is_zimage_diffusers(d):
                opts.append(str(d))
    for folder in ("unet", "diffusion_models"):
        base = Path.home() / "ComfyUI-Shared" / "models" / folder
        if not base.is_dir():
            continue
        for f in sorted(base.glob("*.safetensors"), key=lambda x: x.name.lower()):
            if _single_file_is_zimage(f):
                opts.append(str(f))
    opts.append(DEFAULT_REPO)          # always allow "download from HF"
    return opts


def default_model_value() -> str:
    for opt in model_options():
        if opt != DEFAULT_REPO and ("-q4" in opt or _is_mflux_saved(Path(opt))):
            return opt
    return model_options()[0] if model_options() else DEFAULT_REPO


def find_base_snapshot(exclude: str = "") -> Path | None:
    """A snapshot that has text_encoder/ + vae/ (+ tokenizer) to lend to a finetune."""
    for opt in model_options():
        p = Path(opt)
        if not p.is_dir() or str(p) == exclude:
            continue
        if _is_mflux_saved(p):        # quantized dirs may symlink the same components
            if (p / "text_encoder").is_dir() and (p / "vae").is_dir():
                return p
            continue
        if (p / "text_encoder").is_dir() and (p / "vae").is_dir():
            return p
    return None


# ------------------------------------------------------------- controlnet ---

CONTROL_TYPES = ["canny", "depth", "pose", "hed", "mlsd"]


def controlnet_checkpoints() -> list[str]:
    """Union ControlNet safetensors found under models/MLX/* (or */controlnet)."""
    root = mlx_root()
    found: list[str] = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            for sub in (d, d / "controlnet"):
                if not sub.is_dir():
                    continue
                for f in sorted(sub.glob("*.safetensors")):
                    if "controlnet" in f.name.lower() and str(f) not in found:
                        found.append(str(f))
    return found


def ensure_composite(base_dir: Path, ckpt: Path) -> Path:
    """A dir mflux can load as "base weights + controlnet/" in one go."""
    comp = mlx_root() / f"{base_dir.name}-cn"
    if (comp / "transformer").is_dir() and (comp / "controlnet").is_dir():
        return comp

    comp.mkdir(parents=True, exist_ok=True)
    for name in ("transformer", "text_encoder", "vae", "tokenizer"):
        src, dst = base_dir / name, comp / name
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src, target_is_directory=True)
            except OSError:
                pass
    for f in list(base_dir.glob("*.json")) + list(base_dir.glob(".source")):
        dst = comp / f.name
        if f.is_file() and not dst.exists():
            try:
                dst.symlink_to(f)
            except OSError:
                pass
    cdir = comp / "controlnet"
    cdir.mkdir(exist_ok=True)
    target = cdir / ckpt.name
    if not target.exists():
        try:
            target.symlink_to(ckpt)
        except OSError:
            pass
    print(f"[ZImage-MLX] 组合目录就绪：{comp}", flush=True)
    return comp


# ------------------------------------------------- single-file checkpoints ---


def _prepared_for(src: Path) -> Path | None:
    """Return an already-prepared MLX dir for this checkpoint, if any."""
    src = src.resolve()
    root = mlx_root()
    if root.is_dir():
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if _saved_source(d) == str(src) or _saved_source(d) == str(src.parent / src.name):
                if (d / "transformer").is_dir() or _is_mflux_saved(d):
                    return d
    for name in (f"{src.stem}-q4", f"{src.stem}", f"Z-Image-{src.stem}-q4"):
        cand = root / name
        if cand.is_dir() and ((cand / "transformer").is_dir() or _is_mflux_saved(cand)):
            return cand
    return None


def _run_stream(cmd: list[str], label: str, verbose: bool) -> None:
    print(f"[ZImage-MLX] {label}: {' '.join(str(c) for c in cmd[:8])} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if verbose or line.startswith(("[", "Saved", "Error", "Traceback")):
            print("   " + line[:200], flush=True)
    code = proc.wait()
    if code != 0:
        raise ZImageRunError(f"{label} 失败（退出码 {code}），详见控制台日志。")


def _prepare_from_checkpoint(src: Path, verbose: bool = False) -> Path:
    """Convert a ComfyUI single-file DiT into MLX dirs, once: bf16 then 4-bit."""
    root = mlx_root()
    root.mkdir(parents=True, exist_ok=True)
    bf16_dir = root / src.stem
    q4_dir = root / f"{src.stem}-q4"

    base = find_base_snapshot()
    if base is None:
        raise ZImageRunError(
            "找不到可借用的基础资产快照（需要含 text_encoder/ 与 vae/ 的 Z-Image 目录）。\n"
            f"请先把它放到 {root}/Z-Image-Turbo（或任何下拉里能看到的完整快照）。"
        )

    if not (bf16_dir / "transformer").is_dir():
        _run_stream(
            [sys.executable, str(NODE_DIR / "scripts" / "convert_zimage_checkpoint.py"),
             "--input", str(src), "--out", str(bf16_dir), "--base", str(base)],
            f"转换 {src.name} -> {bf16_dir.name}（约 12GB bf16）",
            verbose,
        )
    else:
        print(f"[ZImage-MLX] 复用已转换的 {bf16_dir}", flush=True)

    if not _is_mflux_saved(q4_dir):
        python = find_python()
        mflux_save = Path(python).parent / "mflux-save"
        if not mflux_save.is_file():
            raise ZImageRunError(f"找不到 mflux-save（应在 {mflux_save}）")
        _run_stream(
            [str(mflux_save), "--model", str(bf16_dir), "--path", str(q4_dir), "--quantize", "4"],
            f"落盘 4bit -> {q4_dir.name}",
            verbose,
        )

    try:
        (q4_dir / ".source").write_text(str(src.resolve()), encoding="utf-8")
    except OSError:
        pass
    return q4_dir


# ------------------------------------------------------------------ nodes ---


class ZImageMlxLoader:
    """Resolve the mflux interpreter + model source, emit a config object."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": (
                    model_options(),
                    {
                        "default": default_model_value(),
                        "tooltip": (
                            "下拉里的本地模型可直接用；unet/ 下的单文件 .safetensors "
                            "会在首次运行时自动转换 + 量化（约 15 分钟，一次性）"
                        ),
                    },
                ),
                "quantization": (
                    QUANTS,
                    {
                        "default": "4",
                        "tooltip": "4=4bit DiT（16GB 机器推荐）；8=8bit；bf16=不量化（需 ~24GB+）",
                    },
                ),
            },
            "optional": {
                "auto_prepare": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "选中 unet/ 单文件时，首次自动转换为 MLX 目录并落盘 4bit"},
                ),
                "low_ram": ("BOOLEAN", {"default": False, "tooltip": "低内存模式：更省内存、更慢"}),
                "cache_limit_gb": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "0=用 mflux 默认；24GB 机器可试 8-12"},
                ),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "把 mflux 全部日志打印到 ComfyUI 控制台"}),
            },
        }

    RETURN_TYPES = ("ZIMAGE_MLX_CONFIG",)
    RETURN_NAMES = ("zimage_mlx_model",)
    FUNCTION = "load"
    CATEGORY = "ZImage MLX"

    def load(
        self,
        model: str,
        quantization: str,
        auto_prepare: bool = True,
        low_ram: bool = False,
        cache_limit_gb: int = 0,
        verbose: bool = False,
    ) -> tuple[dict[str, Any]]:
        model = _as_str(model, DEFAULT_REPO, "model").strip() or DEFAULT_REPO

        # A single-file checkpoint picked from unet/ -- prepare it once, then use the result.
        if model.lower().endswith((".safetensors", ".sft")):
            src = Path(os.path.expanduser(model))
            if not src.is_file():
                raise ZImageRunError(f"选中的单文件不存在：{src}")
            ready = _prepared_for(src)
            if ready is None:
                if not auto_prepare:
                    raise ZImageRunError(
                        f"{src.name} 还是原始单文件，需要先转换。\n"
                        "把 auto_prepare 打开（首次约 15 分钟），或手动跑 "
                        "scripts/convert_zimage_checkpoint.py。"
                    )
                ready = _prepare_from_checkpoint(src, verbose=verbose)
            model = str(ready)
        if quantization not in QUANTS:
            _fix("quantization", f"{quantization!r} -> 4")
            quantization = "4"
        cache_limit_gb = _as_int(cache_limit_gb, 0, 0, 64, "cache_limit_gb")
        _report_fixes()

        # A path must exist; a repo id is downloaded by mflux on first run.
        expanded = Path(os.path.expanduser(model))
        if expanded.is_absolute() or model.startswith("."):
            if not expanded.is_dir():
                raise ZImageRunError(
                    f"模型目录不存在：{expanded}\n"
                    "要么填 HF 仓库名（如 Tongyi-MAI/Z-Image-Turbo）让 mflux 自动下载，"
                    "要么先下载好 diffusers 目录再填路径。"
                )
            model = str(expanded)
            has_files = (expanded / "transformer").is_dir() and (expanded / "text_encoder").is_dir()
            if not has_files:
                raise ZImageRunError(
                    f"{expanded} 看起来不是 diffusers 快照（缺 transformer/ 或 text_encoder/）。"
                )

        python = find_python()
        print(
            f"[ZImage-MLX] v{VERSION} | model={model} | quant={quantization} | python={python}",
            flush=True,
        )
        return (
            {
                "model": model,
                "quantization": quantization,
                "python": python,
                "low_ram": bool(low_ram),
                "cache_limit_gb": cache_limit_gb,
                "verbose": bool(verbose),
            },
        )


class ZImageMlxTextToImage:
    """Z-Image-Turbo text-to-image through mflux (MLX)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "zimage_mlx_model": ("ZIMAGE_MLX_CONFIG",),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Z-Image 中英文都强，可直接写中文"},
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "steps": (
                    "INT",
                    {"default": 9, "min": 1, "max": 50, "step": 1, "tooltip": "蒸馏模型，8-9 步即可；更高基本只增加耗时"},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "ZImage MLX"

    def generate(
        self,
        zimage_mlx_model: dict[str, Any],
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
    ):
        prompt = _as_str(prompt, "", "prompt", allow_numbers=True)
        width = _snap32(_as_int(width, 1024, 256, 1536, "width"), "width")
        height = _snap32(_as_int(height, 1024, 256, 1536, "height"), "height")
        steps = _as_int(steps, 9, 1, 50, "steps")
        seed = _as_int(seed, 42, 0, 2**31 - 1, "seed")
        _report_fixes()

        if not prompt.strip():
            raise ZImageRunError("prompt 为空：Z-Image-Turbo 需要提示词。")

        bar = None
        if comfy_utils is not None:
            try:
                bar = comfy_utils.ProgressBar(steps)
            except Exception:
                bar = None

        def on_progress(step: int, total: int) -> None:
            if bar is not None:
                try:
                    bar.update_absolute(min(step, total), total)
                except Exception:
                    pass

        def should_cancel() -> bool:
            if comfy_mm is None:
                return False
            try:
                return bool(comfy_mm.processing_interrupted())
            except Exception:
                return False

        verbose = bool(zimage_mlx_model.get("verbose"))

        def on_log(line: str) -> None:
            if verbose or line.startswith("[zimage-mlx]"):
                print(line, flush=True)

        with tempfile.TemporaryDirectory(prefix="zimage_mlx_") as tmp:
            out_path = Path(tmp) / f"zimage_{seed}_{width}x{height}.png"
            run_generate(
                python=zimage_mlx_model["python"],
                model=zimage_mlx_model["model"],
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                quantize=zimage_mlx_model.get("quantization", "4"),
                out_path=out_path,
                low_ram=bool(zimage_mlx_model.get("low_ram")),
                cache_limit_gb=int(zimage_mlx_model.get("cache_limit_gb") or 0),
                on_progress=on_progress,
                on_log=on_log,
                should_cancel=should_cancel,
            )
            image = Image.open(out_path)
            image.load()
            return (_pil_to_tensor(image),)


class ZImageMlxControlNet:
    """Z-Image-Turbo + ControlNet Union (canny / depth / pose / hed / mlsd)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        ckpts = controlnet_checkpoints()
        return {
            "required": {
                "zimage_mlx_model": ("ZIMAGE_MLX_CONFIG",),
                "control_image": (
                    "IMAGE",
                    {"tooltip": "预处理好的控制图（Canny/Depth/OpenPose 的输出），不是原图"},
                ),
                "control_type": (
                    CONTROL_TYPES,
                    {"default": "canny", "tooltip": "告诉模型这是哪一类控制信号"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1, "step": 1}),
                "control_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05,
                     "tooltip": "单路控制强度，模型卡推荐 0.65-1.0"},
                ),
                "controlnet_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "全局倍率，作用在所有控制上"},
                ),
            },
            "optional": {
                "controlnet_model": (
                    ckpts or ["(未找到 Union ControlNet)"],
                    {"default": ckpts[0] if ckpts else "(未找到 Union ControlNet)"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "ZImage MLX"

    def generate(
        self,
        zimage_mlx_model: dict[str, Any],
        control_image: torch.Tensor,
        control_type: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        control_strength: float,
        controlnet_strength: float,
        controlnet_model: str = "",
    ):
        prompt = _as_str(prompt, "", "prompt", allow_numbers=True)
        width = _snap32(_as_int(width, 1024, 256, 1536, "width"), "width")
        height = _snap32(_as_int(height, 1024, 256, 1536, "height"), "height")
        steps = _as_int(steps, 8, 1, 50, "steps")
        seed = _as_int(seed, 42, 0, 2**31 - 1, "seed")
        control_strength = _as_float(control_strength, 0.8, 0.0, 1.5, "control_strength")
        controlnet_strength = _as_float(controlnet_strength, 1.0, 0.0, 2.0, "controlnet_strength")
        if control_type not in CONTROL_TYPES:
            _fix("control_type", f"{control_type!r} -> canny")
            control_type = "canny"
        _report_fixes()

        ckpt = Path(os.path.expanduser(str(controlnet_model or "")))
        if not ckpt.is_file():
            avail = controlnet_checkpoints()
            raise ZImageRunError(
                "找不到 Union ControlNet 权重。\n"
                "下载 alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 里的 "
                "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors，\n"
                f"放到 {mlx_root()} 下任意目录（会递归扫描）。"
                + (f"\n当前可用：{avail[:3]}" if avail else "")
            )

        base_dir = Path(zimage_mlx_model["model"])
        if control_image is None:
            raise ZImageRunError("control_image 为空：需要一张预处理过的控制图。")

        bar = None
        if comfy_utils is not None:
            try:
                bar = comfy_utils.ProgressBar(steps)
            except Exception:
                bar = None

        def on_progress(step: int, total: int) -> None:
            if bar is not None:
                try:
                    bar.update_absolute(min(step, total), total)
                except Exception:
                    pass

        def should_cancel() -> bool:
            if comfy_mm is None:
                return False
            try:
                return bool(comfy_mm.processing_interrupted())
            except Exception:
                return False

        verbose = bool(zimage_mlx_model.get("verbose"))

        def on_log(line: str) -> None:
            if verbose or line.startswith("[zimage-mlx]"):
                print(line, flush=True)

        with tempfile.TemporaryDirectory(prefix="zimage_cn_") as tmp:
            tmp_dir = Path(tmp)
            hint = tmp_dir / "control.png"
            _tensor_to_pil(control_image).save(hint)
            comp = ensure_composite(base_dir, ckpt)
            out_path = tmp_dir / f"cn_{seed}_{width}x{height}.png"
            run_controlnet(
                python=zimage_mlx_model["python"],
                model=str(comp),
                controlnet_ckpt=str(ckpt),
                control_image=hint,
                control_type=control_type,
                control_strength=control_strength,
                controlnet_strength=controlnet_strength,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                out_path=out_path,
                on_progress=on_progress,
                on_log=on_log,
                should_cancel=should_cancel,
            )
            image = Image.open(out_path)
            image.load()
            return (_pil_to_tensor(image),)


NODE_CLASS_MAPPINGS = {
    "ZImage MLX Loader": ZImageMlxLoader,
    "ZImage MLX Text to Image": ZImageMlxTextToImage,
    "ZImage MLX ControlNet": ZImageMlxControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImage MLX Loader": "Z-Image-Turbo MLX 模型配置",
    "ZImage MLX Text to Image": "Z-Image-Turbo MLX 文生图",
    "ZImage MLX ControlNet": "Z-Image-Turbo MLX ControlNet",
}

print(f"[ZImage-MLX] nodes v{VERSION} loaded", flush=True)
