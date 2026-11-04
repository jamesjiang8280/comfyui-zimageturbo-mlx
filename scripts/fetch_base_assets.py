#!/usr/bin/env python3
"""Fetch the base Z-Image assets mflux needs (text encoder + VAE + tokenizer).

The DiT is the part people finetune; the other components are shared by every
Z-Image model.  Converting a single-file checkpoint needs them as a `--base`, and
running the base model needs them too.  This pulls just those (~8.3 GB) instead
of the whole 33 GB repo, using hf-mirror with ranged resume -- huggingface_hub
tends to stall or hit Xet 401 on this kind of link.

Usage:
    python3 fetch_base_assets.py                       # -> ~/ComfyUI-Shared/models/MLX/Z-Image-Turbo
    python3 fetch_base_assets.py --dest /some/dir
    HF_ENDPOINT=https://huggingface.co python3 fetch_base_assets.py --no-mirror

After it finishes, `--base <dest>` works for convert_zimage_checkpoint.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = "Tongyi-MAI/Z-Image-Turbo"
WANTED = ("text_encoder/", "vae/", "tokenizer/", "scheduler/")
EXTRA = ("model_index.json", "transformer/config.json")


def api_listing(endpoint: str) -> list[dict]:
    url = f"{endpoint}/api/models/{REPO}?blobs=true"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r).get("siblings", [])
        except Exception as exc:
            print(f"[warn] listing attempt {attempt + 1}: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"无法获取 {REPO} 的文件列表（试试 --no-mirror 或设 HF_ENDPOINT）")


def total_size(url: str) -> int:
    hdr = Path("/tmp/.cr_base")
    for attempt in range(6):
        subprocess.run(["curl", "-sL", "--max-time", "60", "-r", "0-0", "-o", "/dev/null",
                        "-D", str(hdr), url], check=False)
        for line in hdr.read_text(errors="ignore").splitlines():
            if line.lower().startswith("content-range:"):
                return int(line.rsplit("/", 1)[1].strip())
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"读不到文件大小：{url}")


def fetch(url: str, target: Path, expect: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expect:
        print(f"[skip] {target.name} ({expect / 1e9:.2f} GB)", flush=True)
        return
    for attempt in range(6):
        have = target.stat().st_size if target.exists() else 0
        if have > expect:
            target.unlink()
            have = 0
        print(f"[get ] {target.name} {have / 1e9:.2f}/{expect / 1e9:.2f} GB", flush=True)
        subprocess.run(["curl", "-sL", "--fail", "--max-time", "3600", "-C", "-",
                        "-o", str(target), url], check=False)
        if target.exists() and target.stat().st_size == expect:
            print(f"[done] {target.name}", flush=True)
            return
        time.sleep(min(30, 5 * attempt))
    raise SystemExit(f"[fail] {target.name}: {target.stat().st_size if target.exists() else 0} != {expect}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path.home() / "ComfyUI-Shared" / "models" / "MLX" / "Z-Image-Turbo"))
    ap.add_argument("--no-mirror", action="store_true", help="use huggingface.co instead of hf-mirror.com")
    args = ap.parse_args()

    endpoint = "https://huggingface.co" if args.no_mirror else "https://hf-mirror.com"
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    files = [s for s in api_listing(endpoint)
             if s.get("size") and (s["rfilename"].startswith(WANTED) or s["rfilename"] in EXTRA)]
    total = sum(s["size"] for s in files)
    print(f"[plan] {len(files)} 个文件 / {total / 1e9:.2f} GB -> {dest}", flush=True)

    for s in sorted(files, key=lambda x: x["rfilename"]):
        name, size = s["rfilename"], s["size"]
        fetch(f"{endpoint}/{REPO}/resolve/main/{name}", dest / name, size)

    print(f"[ok] base 资产就绪：{dest}", flush=True)
    print(f"     下一步：python3 convert_zimage_checkpoint.py --input <ckpt> --out <dir> --base {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
