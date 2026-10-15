#!/usr/bin/env python
"""从真实 KG 文件提取"型号注册表"。

数据来源只有一处：传入的 KG 目录（默认是原仓库的 random_data/test1/kg）。
产出的是**元信息**（型号 id、每个图的节点数/前缀长度/校验结果），
不生成任何训练数据、不猜测任何缺失字段。

用法：
    python scripts/02_build_model_registry.py                 # 扫描原仓库 Demo 数据
    python scripts/02_build_model_registry.py --kg_dir <dir> --out <path.json>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.config.settings import load_settings            # noqa: E402
from assemmate_ext.kg.io import read_documents                     # noqa: E402
from assemmate_ext.kg.legacy import load_legacy_dir                # noqa: E402
from assemmate_ext.kg.model_registry import ModelRegistry          # noqa: E402
from assemmate_ext.kg.validate import validate_document            # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout          # noqa: E402
from assemmate_ext.utils.jsonio import write_json                  # noqa: E402

setup_utf8_stdout()


def main() -> int:
    parser = argparse.ArgumentParser(description="构建产品型号注册表")
    parser.add_argument("--config", default=None)
    parser.add_argument("--kg_dir", default=None, help="KG 目录（默认取原仓库 Demo 数据）")
    parser.add_argument("--workspace_root", default=None)
    parser.add_argument("--out", default=None, help="输出 JSON 路径（默认 data/processed/model_registry.json）")
    args = parser.parse_args()

    settings = load_settings(config_path=args.config, workspace_root=args.workspace_root)
    paths = settings.paths

    kg_dir = Path(args.kg_dir) if args.kg_dir else paths.demo_kg_dir
    if not kg_dir or not Path(kg_dir).is_dir():
        print(f"[FAIL] KG 目录不存在: {kg_dir}", file=sys.stderr)
        return 1

    # 同时支持"旧格式"与"统一 Schema"两种输入，自动判别
    legacy = load_legacy_dir(kg_dir)
    unified = []
    if not legacy:
        unified = read_documents(kg_dir)
    documents = [g.document for g in legacy] + unified

    if not documents:
        print(f"[FAIL] {kg_dir} 下未解析出任何图谱文档", file=sys.stderr)
        return 1

    registry = ModelRegistry.from_documents(documents)

    summary = []
    for g in legacy:
        vr = validate_document(g.document)
        summary.append({
            "file": g.path.name,
            "doc_id": g.doc_id,
            "model_id": g.model_guess,
            "n_entities": len(g.entity_names),
            "n_relations": len(g.relation_names),
            "n_prefix_tokens": g.n_prefix_tokens,
            "has_triples": bool(g.document.triples),
            "validation_ok": vr.ok,
            "n_errors": len(vr.errors),
            "n_warnings": len(vr.warnings),
        })

    out_path = Path(args.out) if args.out else (Path(paths.processed_data_dir) / "model_registry.json")
    payload = {
        "source_dir": str(kg_dir),
        "source_format": "legacy" if legacy else "unified_schema",
        "n_documents": len(documents),
        "n_models": len(registry),
        "model_ids": registry.model_ids(),
        "registry": registry.to_dict(),
        "per_document": summary,
        "note": (
            "型号 id 由节点名前缀推断（kg/model_registry.guess_model_id），"
            "属启发式结果而非人工标注；component_types 未标注的型号保持为空。"
        ),
    }
    write_json(out_path, payload)

    print(f"扫描目录   : {kg_dir}")
    print(f"文档数     : {len(documents)}")
    print(f"识别型号数 : {len(registry)}")
    for mid in registry.model_ids():
        model = registry.get(mid)
        n_comp = len(model.component_types) if model else 0
        print(f"  - {mid:<18} 允许部件类型 {n_comp} 个")
    print(f"已写入     : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
