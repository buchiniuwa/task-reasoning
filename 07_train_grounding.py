#!/usr/bin/env python
"""实体接地训练入口（N05，自举 sys.path）。

用法：

    # 过拟合健全性检查（不是泛化性能！）
    python scripts/07_train_grounding.py --overfit 30 --epochs 40

    # 小规模留出评估（按 group 切分，防泄漏）
    python scripts/07_train_grounding.py --epochs 30 --val_ratio 0.3

    # 消融：关掉 MRCG
    python scripts/07_train_grounding.py --no_mrcg --out_dir outputs/ablation_no_mrcg

    # 消融：关掉型号掩码（对照"仅掩码是免费分数"这一质疑）
    python scripts/07_train_grounding.py --no_model_mask --out_dir outputs/ablation_no_mask

说明：默认使用**假文本编码器**（确定性哈希），因此无需下载 BGE 即可跑通；
真实 BGE 需要自备本地权重并用 `--text_encoder bge --bge_model <路径>`。
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.training.train_grounding import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
