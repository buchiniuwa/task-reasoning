#!/usr/bin/env bash
# AssemMate 基线复现入口（Git Bash / Linux 便捷封装）
#
# 真正的逻辑在 src/assemmate_ext/baselines/runner.py，由 01_run_baseline.py 自举导入。
# 之所以不在这里设 PYTHONPATH：Git Bash 的 POSIX 路径（/d/...）Windows 版 Python
# 无法识别，跨平台容易踩坑，交给 Python 侧注入 sys.path 最稳。
#
# 用法：
#   bash scripts/01_baseline_reproduce.sh validate       # 只做环境与数据自检（不需要 torch）
#   bash scripts/01_baseline_reproduce.sh instrumented   # 逐题推理 + 落盘 + 统一评估
#   bash scripts/01_baseline_reproduce.sh original       # 直接调用原版 evaluate
#   bash scripts/01_baseline_reproduce.sh smoke          # 只跑前 5 条（冒烟测试）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-instrumented}"
shift || true

case "${MODE}" in
  validate)
    exec python "${SCRIPT_DIR}/01_run_baseline.py" --validate-only "$@"
    ;;
  original)
    exec python "${SCRIPT_DIR}/01_run_baseline.py" --mode original "$@"
    ;;
  instrumented)
    exec python "${SCRIPT_DIR}/01_run_baseline.py" --mode instrumented "$@"
    ;;
  smoke)
    exec python "${SCRIPT_DIR}/01_run_baseline.py" --mode instrumented --limit 5 "$@"
    ;;
  *)
    echo "未知模式: ${MODE}" >&2
    echo "可选: validate | instrumented | original | smoke" >&2
    exit 2
    ;;
esac
