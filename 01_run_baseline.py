#!/usr/bin/env python
"""基线复现入口（跨平台，自举 sys.path）。

用法（参数与原版 infer_and_eval.py 兼容，并额外支持 --config/--bert_model 等）：

    # 只做环境与数据自检（不需要 torch / 模型权重）
    python scripts/01_run_baseline.py --validate-only

    # 逐题推理 + 落盘 + 统一评估
    python scripts/01_run_baseline.py \
        --base_model D:/models/llama3-8b-4bit \
        --projector_ckpt_path D:/models/projector_weights.pt \
        --bert_model D:/models/bert-base-uncased

    # 直接调用原版 evaluate（验证 BERT 路径补丁确实生效）
    python scripts/01_run_baseline.py --mode original ...

说明：本脚本只做 sys.path 注入并转发参数，真正的逻辑全部在
src/assemmate_ext/baselines/runner.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.baselines.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
