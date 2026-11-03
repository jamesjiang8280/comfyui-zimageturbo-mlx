#!/usr/bin/env bash
# Prepare the Z-Image-Turbo MLX node.
#
#   bash scripts/setup_env.sh            # probe the mflux interpreter, print status
#   bash scripts/setup_env.sh --weights  # ...and download Tongyi-MAI/Z-Image-Turbo (~33 GB)
#
# Nothing is installed into ComfyUI's own environment: mflux already lives in the
# sibling Qwen-Image MLX venv (or wherever ZIMAGE_MLX_PYTHON points).

set -euo pipefail

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ZIMAGE_MLX_MODELS_DIR:-$HOME/ComfyUI-Shared/models/MLX}"
SNAPSHOT="$MODELS_DIR/Z-Image-Turbo"
REPO="Tongyi-MAI/Z-Image-Turbo"
WANT_WEIGHTS=0

for arg in "$@"; do
  case "$arg" in
    --weights) WANT_WEIGHTS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "只支持 macOS（MLX 需要 Apple Silicon）。"
[ "$(uname -m)" = "arm64" ] || die "需要 Apple Silicon (arm64)，当前 $(uname -m)。"

# ------------------------------------------------------------- interpreter --
PY="${ZIMAGE_MLX_PYTHON:-}"
if [ -z "$PY" ]; then
  SIBLING="$NODE_DIR/../comfyui-qwen21-mlx/.venv/bin/python"
  if [ -x "$SIBLING" ]; then PY="$SIBLING"; fi
fi
if [ -z "$PY" ]; then
  PY="$(command -v python3 || true)"
fi
[ -n "$PY" ] && [ -x "$PY" ] || die "找不到 python3。"

"$PY" - <<'EOF' || die "该 Python 里没有 mflux 的 z-image CLI。请装 mflux 或用 ZIMAGE_MLX_PYTHON 指定。"
import mlx.core as mx, importlib.metadata as md
import mflux.models.z_image.cli.z_image_turbo_generate  # noqa: F401
print("mflux", md.version("mflux"), "| mlx", md.version("mlx"), "| device", mx.default_device())
EOF

say "Python: $PY"

# -------------------------------------------------------------- download ----
if [ "$WANT_WEIGHTS" = "1" ]; then
  if [ -d "$SNAPSHOT/transformer" ]; then
    say "权重已存在：$SNAPSHOT"
  else
    say "下载 $REPO -> $SNAPSHOT（约 33GB fp32，mflux 加载时再量化）"
    mkdir -p "$SNAPSHOT"
    "$PY" -m huggingface_hub.commands.huggingface_cli download "$REPO" \
        --local-dir "$SNAPSHOT" --max-workers 4
  fi
fi

cat <<EOF

$(say "状态")
  Python : $PY
  模型   : $SNAPSHOT $([ -d "$SNAPSHOT/transformer" ] && echo '(就绪)' || echo '(缺失，加 --weights 下载)')

冒烟测试（不经过 ComfyUI，直接用 mflux）：
  "$PY" -m mflux.models.z_image.cli.z_image_turbo_generate \\
      --model "$SNAPSHOT" --quantize 4 --steps 9 --seed 42 \\
      --width 768 --height 768 --prompt "a lighthouse at dusk" --output /tmp/zimage_test.png

然后把本目录整个放进 ComfyUI/custom_nodes/ 并重启 ComfyUI。
EOF
