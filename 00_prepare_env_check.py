#!/usr/bin/env python
"""环境自检 —— 只读检查，不安装依赖、不下载模型、不运行训练。

用法：
    python scripts/00_prepare_env_check.py
    python scripts/00_prepare_env_check.py --config configs/baseline.yaml
    python scripts/00_prepare_env_check.py --json outputs/preflight.json

退出码：
    0 全部就绪 | 2 仅可做静态检查与单元测试 | 1 存在阻塞问题
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 脚本自举：让 `python scripts/xx.py` 直接可用，无需 pip install -e .
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.config.settings import load_settings          # noqa: E402
from assemmate_ext.preflight import run_preflight                 # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout         # noqa: E402
from assemmate_ext.utils.jsonio import write_json                 # noqa: E402

setup_utf8_stdout()


def main() -> int:
    parser = argparse.ArgumentParser(description="AssemMate 扩展框架环境自检（只读）")
    parser.add_argument("--config", default=None, help="配置文件路径（默认自动寻找 configs/baseline.yaml）")
    parser.add_argument("--struct_dir", default=None, help="KG 目录（默认取原仓库 random_data/test1/kg）")
    parser.add_argument("--qa_dir", default=None, help="QA 目录")
    parser.add_argument("--workspace_root", default=None, help="工作区根目录")
    parser.add_argument("--json", dest="json_out", default=None, help="把自检报告写到该 JSON 路径")
    parser.add_argument("--no-data-check", action="store_true", help="跳过数据语义解析（大数据集可提速）")
    args = parser.parse_args()

    settings = load_settings(config_path=args.config, workspace_root=args.workspace_root)
    report = run_preflight(
        settings,
        kg_dir=args.struct_dir,
        qa_dir=args.qa_dir,
        deep_data_check=not args.no_data_check,
    )
    print(report.format_text())

    if args.json_out:
        out = write_json(Path(args.json_out), report.to_dict())
        print(f"自检报告已写入: {out}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
