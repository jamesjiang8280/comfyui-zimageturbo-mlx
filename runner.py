"""Subprocess bridge to mflux's Z-Image-Turbo CLI (MLX on Apple Silicon).

ComfyUI stays on the Python side; the 6B S3-DiT runs inside mflux's own CLI
(``mflux.models.z_image.cli.z_image_turbo_generate``) in an isolated interpreter,
which is the same pattern the Qwen-Image-2.1 MLX node in this install uses.

Why a subprocess instead of importing mflux here:
  * mflux pins its own mlx / transformers / huggingface_hub versions, and ComfyUI
    pins different ones -- importing both into one process is how you get
    "OperatorNotImplemented" at 3am;
  * a crashed/failed generation then cannot take ComfyUI down with it;
  * the CLI is the surface mflux's own tests exercise.

Note: on Z-Image Turbo, mflux *ignores* ``--guidance`` (forced 0.0) and
``--negative-prompt`` (CFG disabled) by design -- it is a guidance-distilled
model.  Both options are therefore not exposed by the node.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

MODULE = "mflux.models.z_image.cli.z_image_turbo_generate"

# tqdm (stderr): "  6/9 [00:05<00:02,  1.20it/s]"  or  " 67%|####  | 6/9 [..."
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")

RATE_RE = re.compile(r"(\d+\.\d+\s*(?:s/it|it/s))")
# mflux's non-step phases, so a long load/reload doesn't look frozen
PHASE_RE = re.compile(r"(Loading|Encod\w*|Decod\w*|text encoder|VAE)")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class ZImageRunError(RuntimeError):
    """Raised when mflux fails (missing weights, OOM, bad args...)."""


def find_python() -> str:
    """Locate an interpreter that can import mflux's z-image CLI.

    Order: explicit override -> the sibling Qwen21 MLX venv (already provisioned
    and proven on this machine) -> whatever interpreter is running ComfyUI.
    """
    import sys

    candidates: list[str] = []
    env = os.environ.get("ZIMAGE_MLX_PYTHON", "").strip()
    if env:
        candidates.append(os.path.expanduser(env))

    # Both the raw and the symlink-resolved location, so a symlinked install
    # still finds its sibling venv.
    here = Path(__file__)
    parents = {here.parent.parent, here.resolve().parent.parent}
    for parent in parents:
        candidates.append(str(parent / "comfyui-qwen21-mlx" / ".venv" / "bin" / "python"))

    # Dedicated mflux venvs: the usual answer when ComfyUI's own env has to stay
    # clean (mflux pins its own transformers/mlx; ComfyUI pins different ones).
    home = Path.home()
    for venv in (home / "mflux-venv", home / "ComfyUI-Shared" / "mflux-venv"):
        candidates.append(str(venv / "bin" / "python"))

    candidates.append(sys.executable)

    seen: set[str] = set()
    for python in candidates:
        if python in seen:
            continue
        seen.add(python)
        if not Path(python).is_file():
            continue
        try:
            probe = subprocess.run(
                [python, "-c", "import mflux.models.z_image.cli.z_image_turbo_generate"],
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return python

    raise ZImageRunError(
        "找不到能运行 mflux Z-Image 的 Python 环境。\n"
        "请设置环境变量 ZIMAGE_MLX_PYTHON 指向装有 mflux 的 python，例如：\n"
        "  export ZIMAGE_MLX_PYTHON=$HOME/ComfyUI-Installs/ComfyUI/ComfyUI/"
        "custom_nodes/comfyui-qwen21-mlx/.venv/bin/python"
    )


def build_command(
    *,
    python: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    quantize: str,
    out_path: Path,
    low_ram: bool = False,
    cache_limit_gb: int = 0,
) -> list[str]:
    cmd = [
        python, "-m", MODULE,
        "--model", model,
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--seed", str(seed),
        "--output", str(out_path),
    ]
    if quantize and quantize != "bf16":
        cmd += ["--quantize", quantize]
    if low_ram:
        cmd.append("--low-ram")
    if cache_limit_gb and cache_limit_gb > 0:
        cmd += ["--mlx-cache-limit-gb", str(cache_limit_gb)]
    return cmd


def stream_process(
    cmd: list[str],
    *,
    label: str,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    out_path: Path | None = None,
) -> None:
    """Run a command, forward its output, and turn tqdm redraws into step lines."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONIOENCODING"] = "utf-8"

    if on_log:
        on_log("[zimage-mlx] " + " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=0, env=env)
    tail: list[str] = []
    pending = b""
    last_step = -1
    announced: set[str] = set()
    started = time.time()

    def handle(line: str) -> None:
        nonlocal last_step
        line = ANSI_RE.sub("", line).strip()
        if not line:
            return
        tail.append(line)
        del tail[:-40]
        if on_log:
            on_log(line)

        m = STEP_RE.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            if on_progress:
                on_progress(step, total)
            if step != last_step and step > 0:
                last_step = step
                rate = RATE_RE.search(line)
                suffix = f"  {rate.group(1).strip()}" if rate else ""
                pct = f" ({step * 100 // total}%)" if total else ""
                print(f"[zimage-mlx] 采样 {step}/{total}{pct}{suffix}"
                      f"  已用 {time.time() - started:.0f}s", flush=True)
            return

        ph = PHASE_RE.search(line)
        if ph and ph.group(1) not in announced:
            announced.add(ph.group(1))
            print(f"[zimage-mlx] {line[:120]}", flush=True)

    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            pending += chunk
            while True:
                cut = min((i for i in (pending.find(b"\n"), pending.find(b"\r")) if i >= 0),
                          default=-1)
                if cut < 0:
                    break
                piece, pending = pending[:cut], pending[cut + 1:]
                handle(piece.decode("utf-8", errors="replace"))
            if should_cancel and should_cancel():
                proc.kill()
                raise ZImageRunError("已取消（ComfyUI 中断）。")
        if pending:
            handle(pending.decode("utf-8", errors="replace"))
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if code != 0:
        raise ZImageRunError(f"{label} 退出码 {code}。最近输出：\n" + "\n".join(tail[-15:]))
    if out_path is not None and not out_path.is_file():
        raise ZImageRunError(
            f"{label} 正常退出但没有产出图片。最近输出：\n" + "\n".join(tail[-15:])
        )
    print(f"[zimage-mlx] 完成，用时 {time.time() - started:.1f}s"
          + (f" -> {out_path.name}" if out_path else ""), flush=True)


def run_controlnet(
    *,
    python: str,
    model: str,
    controlnet_ckpt: str,
    control_image: Path,
    control_type: str,
    control_strength: float,
    controlnet_strength: float,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = 42,
    out_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """ControlNet Union run through scripts/zimage_controlnet_run.py."""
    script = Path(__file__).resolve().parent / "scripts" / "zimage_controlnet_run.py"
    if not script.is_file():
        raise ZImageRunError(f"找不到 ControlNet 驱动脚本：{script}")

    print(f"[zimage-mlx] 开始生成（ControlNet {control_type}）{width}x{height} "
          f"steps={steps} seed={seed} strength={control_strength}", flush=True)
    cmd = [
        python, str(script),
        "--model", str(model),
        "--control-type", str(control_type),
        "--control-image", str(control_image),
        "--control-strength", str(control_strength),
        "--controlnet-strength", str(controlnet_strength),
        "--prompt", prompt,
        "--width", str(width), "--height", str(height),
        "--steps", str(steps), "--seed", str(seed),
        "--output", str(out_path),
    ]
    stream_process(cmd, label="mflux controlnet", on_progress=on_progress,
                   on_log=on_log, should_cancel=should_cancel, out_path=out_path)
    return out_path


def run_generate(
    *,
    python: str,
    model: str,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    seed: int = 0,
    quantize: str = "4",
    out_path: Path,
    low_ram: bool = False,
    cache_limit_gb: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Run one generation; returns the written PNG path."""
    cmd = build_command(
        python=python,
        model=model,
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        quantize=quantize,
        out_path=out_path,
        low_ram=low_ram,
        cache_limit_gb=cache_limit_gb,
    )

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # tqdm only emits step bars when it believes it has a terminal; ComfyUI's
    # console is not one, so force the bar on (it is our only progress signal).
    env["PYTHONIOENCODING"] = "utf-8"

    if on_log:
        on_log("[zimage-mlx] " + " ".join(cmd))

    print(
        f"[zimage-mlx] 开始生成 {width}x{height} steps={steps} seed={seed} "
        f"quant={quantize} model={Path(model).name}",
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    tail: list[str] = []
    pending = b""
    code: int | None = None
    last_step = -1
    announced: set[str] = set()
    started = time.time()

    def handle(line: str) -> None:
        nonlocal last_step
        line = ANSI_RE.sub("", line).strip()
        if not line:
            return
        tail.append(line)
        del tail[:-40]
        if on_log:
            on_log(line)

        m = STEP_RE.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            if on_progress:
                on_progress(step, total)
            if step != last_step and step > 0:   # tqdm redraws, and step 0 is just its first frame
                last_step = step
                rate = RATE_RE.search(line)
                suffix = f"  {rate.group(1).strip()}" if rate else ""
                pct = f" ({step * 100 // total}%)" if total else ""
                print(f"[zimage-mlx] 采样 {step}/{total}{pct}{suffix}"
                      f"  已用 {time.time() - started:.0f}s", flush=True)
            return

        # non-step phases (load / encode / decode): show once, so a long load isn't silent
        ph = PHASE_RE.search(line)
        if ph and ph.group(1) not in announced:
            announced.add(ph.group(1))
            print(f"[zimage-mlx] {line[:120]}", flush=True)

    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            pending += chunk
            # tqdm redraws with \r, real output uses \n -> treat both as breaks.
            while True:
                cut = min(
                    (i for i in (pending.find(b"\n"), pending.find(b"\r")) if i >= 0),
                    default=-1,
                )
                if cut < 0:
                    break
                piece, pending = pending[:cut], pending[cut + 1:]
                handle(piece.decode("utf-8", errors="replace"))
            if should_cancel and should_cancel():
                proc.kill()
                raise ZImageRunError("已取消（ComfyUI 中断）。")
        if pending:
            handle(pending.decode("utf-8", errors="replace"))
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if code != 0:
        raise ZImageRunError(
            f"mflux 退出码 {code}。最近输出：\n" + "\n".join(tail[-15:])
        )
    if not out_path.is_file():
        raise ZImageRunError(
            "mflux 正常退出但没有产出图片。最近输出：\n" + "\n".join(tail[-15:])
        )
    print(f"[zimage-mlx] 完成，用时 {time.time() - started:.1f}s -> {out_path.name}", flush=True)
    return out_path
