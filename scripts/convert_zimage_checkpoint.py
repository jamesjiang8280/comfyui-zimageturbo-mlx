#!/usr/bin/env python3
"""Convert a ComfyUI-style single-file Z-Image checkpoint into an mflux/MLX tree.

Why: mflux loads a **diffusers directory** (``transformer/ text_encoder/ vae/
tokenizer/``) and looks up weights by diffusers key names.  Community finetunes
ship as one ComfyUI-format ``.safetensors`` whose keys are
``model.diffusion_model.<comfy name>`` and whose attention projections are fused
into a single ``qkv``.  This script rewrites keys and splits the fusion so mflux's
own ``ZImageWeightMapping`` matches every tensor.

Only the DiT comes from your file.  mflux still needs the Qwen3-4B text encoder,
the FLUX AE and the tokenizer -- either point ``--base`` at an existing Z-Image
snapshot to reuse them, or download them once.

Commands
--------
  # 1) check the key mapping against mflux's expectations (header only, no data read)
  python3 convert_zimage_checkpoint.py --input model.safetensors --check

  # 2) convert for real (needs a complete file; ~12 GB bf16 output)
  python3 convert_zimage_checkpoint.py --input model.safetensors \\
      --out ~/ComfyUI-Shared/models/MLX/Z-Image-Turbo-moody \\
      --base ~/ComfyUI-Shared/models/MLX/Z-Image-Turbo
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PREFIX = "model.diffusion_model."

# comfy name -> diffusers name that mflux's ZImageWeightMapping expects
RENAMES = {
    "x_embedder": "all_x_embedder.2-1",
    "final_layer": "all_final_layer.2-1",
}
# comfy name -> diffusers name (attention internals)
ATTR_RENAMES = {
    "attention.q_norm": "attention.norm_q",
    "attention.k_norm": "attention.norm_k",
    "attention.out": "attention.to_out.0",
}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def completeness(path: Path, hdr: dict) -> tuple[int, int]:
    """(actual file size, size the header says it needs)."""
    data_start = 8 + len(json.dumps(hdr, separators=(",", ":")))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    # data_start is approximate (separators differ); recompute exactly below
    with open(path, "rb") as f:
        raw_len = struct.unpack("<Q", f.read(8))[0]
    return path.stat().st_size, 8 + raw_len + end


def convert_keys(hdr: dict) -> dict[str, dict]:
    """comfy key -> {diffusers_key: {shape, dtype, src, qkv}}.

    ``src`` is the exact key inside the source file; ``qkv`` is 0/1/2 for the
    slices of a fused qkv tensor, else None.  Keeping them here means the writer
    never has to reverse-engineer the rename table again.
    """
    out: dict[str, dict] = {}
    for key, meta in hdr.items():
        if key == "__metadata__":
            continue
        shape = meta["shape"]
        k = key.replace(PREFIX, "", 1) if key.startswith(PREFIX) else key
        for src, dst in RENAMES.items():
            if k == src or k.startswith(src + "."):
                k = dst + k[len(src):]
                break
        for src, dst in ATTR_RENAMES.items():
            if ("." + src + ".") in ("." + k):
                k = k.replace("." + src + ".", "." + dst + ".")

        if k.endswith("attention.qkv.weight"):
            stem = k[: -len("qkv.weight")]
            if len(shape) != 2 or shape[0] % 3:
                raise SystemExit(f"fused qkv shape not divisible by 3: {k} {shape}")
            part = shape[0] // 3
            for slot, name in enumerate(("to_q", "to_k", "to_v")):
                out[f"{stem}{name}.weight"] = {
                    "shape": [part, shape[1]], "dtype": meta["dtype"],
                    "src": key, "qkv": slot,
                }
        else:
            out[k] = {"shape": shape, "dtype": meta["dtype"], "src": key, "qkv": None}
    return out


def flatten_mlx(obj, prefix: str = "") -> dict[str, list[int]]:
    """Walk mflux's nested parameter tree (dict / list / array) into {'a.b.c': shape}."""
    out: dict[str, list[int]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_mlx(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_mlx(v, f"{prefix}.{i}" if prefix else str(i)))
    elif hasattr(obj, "shape"):
        out[prefix] = list(obj.shape)
    return out


def check(hdr: dict) -> int:
    """Validate the converted key set against mflux's own mapping + module tree."""
    import re as _re

    from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
    from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping

    converted = convert_keys(hdr)
    params = flatten_mlx(ZImageTransformer().parameters())
    targets = ZImageWeightMapping.get_transformer_mapping()

    # from_pattern -> (regex, to_pattern); placeholders are positional
    compiled = []
    for t in targets:
        pat = t.from_pattern[0]
        rx = _re.escape(pat)
        rx = rx.replace(r"\{layer\}", r"(\d+)").replace(r"\{n\}", r"(\d+)")
        compiled.append((_re.compile("^" + rx + "$"), pat, t.to_pattern))

    unmatched, badshape, used = [], [], set()
    for key, meta in converted.items():
        hit = None
        for rx, pat, to in compiled:
            m = rx.match(key)
            if m:
                hit = (m.groups(), to)
                break
        if hit is None:
            unmatched.append(key)          # mflux has no rule for this tensor
            continue
        groups, to = hit
        tgt = to
        for i, g in enumerate(groups):
            tgt = tgt.replace("{layer}" if i == 0 else "{n}", g, 1)
        # to_pattern may use {layer}/{n} in either slot; resolve leftovers with groups
        if "{" in tgt:
            tgt = tgt.format(layer=groups[0] if groups else 0, n=groups[-1] if groups else 0)
        real = params.get(tgt)
        if real is None:
            unmatched.append(f"{key} -> {tgt}")
        elif list(real) != list(meta["shape"]):
            badshape.append((key, meta["shape"], list(real)))
        else:
            used.add(tgt)

    unfilled = [k for k in params if k not in used]
    print(f"源张量(展开 qkv)   : {len(converted)}")
    print(f"模型参数            : {len(params)}")
    print(f"无对应映射规则      : {len(unmatched)}")
    for k in unmatched[:8]:
        print("   !", k)
    print(f"形状不一致          : {len(badshape)}")
    for k, want, got in badshape[:8]:
        print(f"   ! {k}: file {want} vs model {got}")
    print(f"模型里没被填上的参数: {len(unfilled)}")
    for k in unfilled[:8]:
        print("   !", k)
    ok = not unmatched and not badshape and not unfilled
    print("\nRESULT:", "OK - 每个张量都能落到模型参数上，且形状一致" if ok else "FAIL")
    return 0 if ok else 1


def write(hdr: dict, src: Path, out: Path, base: Path | None) -> int:
    """Convert + save in shards, so peak RAM stays ~one shard (24 GB Macs)."""
    import gc
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    have, need = completeness(src, hdr)
    if have < need:
        raise SystemExit(
            f"输入文件不完整：{have/1e9:.2f} GB / 需要 {need/1e9:.2f} GB。\n"
            "safetensors 会直接报 'incomplete metadata' —— 请重新下载完整文件。"
        )

    tdir = out / "transformer"
    tdir.mkdir(parents=True, exist_ok=True)

    converted = convert_keys(hdr)           # diffusers_name -> {shape, dtype, src, qkv}
    names = list(converted)

    n_shards = 3
    per = (len(names) + n_shards - 1) // n_shards
    weight_map: dict[str, str] = {}
    total_bytes = 0

    with safe_open(str(src), framework="pt") as f:
        for si in range(n_shards):
            chunk = names[si * per:(si + 1) * per]
            if not chunk:
                continue
            shard_name = f"diffusion_pytorch_model-{si+1:05d}-of-{n_shards:05d}.safetensors"
            payload: dict[str, "torch.Tensor"] = {}
            for key in chunk:
                meta = converted[key]
                t = f.get_tensor(meta["src"])
                qi = meta["qkv"]
                if qi is not None:
                    part = t.shape[0] // 3
                    t = t[qi * part:(qi + 1) * part]
                payload[key] = t.to(torch.bfloat16).contiguous()
                weight_map[key] = shard_name
            dest = tdir / shard_name
            save_file(payload, str(dest), metadata={"format": "pt"})
            total_bytes += dest.stat().st_size
            print(f"[write] {shard_name} {len(payload)} tensors "
                  f"{dest.stat().st_size/1e9:.2f} GB", flush=True)
            del payload
            gc.collect()

    (tdir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2))

    if base and (base / "transformer" / "config.json").is_file():
        (tdir / "config.json").write_text((base / "transformer" / "config.json").read_text())

    if base:
        for sub in ("text_encoder", "vae", "tokenizer", "scheduler"):
            s_, d_ = base / sub, out / sub
            if s_.is_dir() and not d_.exists():
                print(f"[link] {sub} <- {s_}", flush=True)
                try:
                    d_.symlink_to(s_, target_is_directory=True)
                except OSError:
                    shutil.copytree(s_, d_)
        mi = base / "model_index.json"
        if mi.is_file() and not (out / "model_index.json").exists():
            shutil.copy2(mi, out / "model_index.json")
    else:
        print("[warn] 没有 --base：还需自行准备 text_encoder/ vae/ tokenizer/ scheduler/")

    print(f"[ok] 转换完成: {out}  总计 {total_bytes/1e9:.2f} GB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out")
    ap.add_argument("--base")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    src = Path(a.input).expanduser()
    hdr = read_header(src)
    have, need = completeness(src, hdr)
    print(f"[info] 文件 {have/1e9:.2f} GB / 头部声明 {need/1e9:.2f} GB "
          f"{'(完整)' if have >= need else '(不完整!)'}")

    if a.check:
        rc = check(hdr)
        if have < need:
            print("\n注意：文件本身不完整，但 key 映射的校验只依赖头部，结果仍然有效。")
        return rc

    if not a.out:
        raise SystemExit("--out 必填（或用 --check 只做校验）")
    return write(hdr, src, Path(a.out).expanduser(), Path(a.base).expanduser() if a.base else None)


if __name__ == "__main__":
    sys.exit(main())
