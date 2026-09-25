#!/usr/bin/env bash
set -euo pipefail
viewer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
viewer_python="${VIEWER_PYTHON:-$viewer_root/.venv/bin/python}"
if [[ ! -x "$viewer_python" ]]; then
  printf '%s\n' '查看器 Python 环境不存在；请安装 .venv，或用 VIEWER_PYTHON 指定 Python 3.10+ 的绝对路径。' >&2
  exit 1
fi
exec "$viewer_python" -m streamlit run "$viewer_root/app.py" \
  --server.port "${VIEWER_PORT:-8502}" \
  --server.address "${VIEWER_HOST:-127.0.0.1}" \
  --server.headless true --browser.gatherUsageStats false "$@"
