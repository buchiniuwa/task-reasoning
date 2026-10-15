#!/usr/bin/env python
"""把两份反事实数据集转成接地实例，并跑
① 信息可识别性检查（第三阶段） ② 强基线输入公平性检查（交付物 5）。

同时也验证**第一阶段的信息通路修复**：实例必须带上
`full_instruction` / `operation_hints` / `operation_candidates` / `candidate_evidence`，
而且模型可见载荷里**不得**出现 gold 字段。

用法：
    python scripts/14_audit_grounding_inputs.py
    python scripts/14_audit_grounding_inputs.py --write-observable outputs/observable.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.builders import Lexicon                     # noqa: E402
from assemmate_ext.dataset.counterfactual import CounterfactualDataset  # noqa: E402
from assemmate_ext.grounding.audit import (                            # noqa: E402
    check_identifiability, check_input_fairness, visible_signature,
)
from assemmate_ext.grounding.dataset import (                          # noqa: E402
    build_counterfactual_instances,
)
from assemmate_ext.utils.console import setup_utf8_stdout              # noqa: E402
from assemmate_ext.utils.jsonio import write_json                      # noqa: E402

setup_utf8_stdout()

SOURCES = [
    ("data/synthetic/counterfactual_connector.json",
     "data/lexicon/counterfactual_lexicon.json", "完整证据三族"),
    ("data/synthetic/counterfactual_partial.json",
     "data/lexicon/counterfactual_partial_lexicon.json", "部分可观测四族"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="接地输入审计")
    parser.add_argument("--out", default="outputs/grounding_input_audit.json")
    parser.add_argument("--write-observable", default=None,
                        help="把模型可见输入（不含 gold）落盘，便于人工核对")
    args = parser.parse_args()

    all_instances: List[Any] = []
    documents: Dict[str, Any] = {}
    print("=" * 78)
    print("接地实例构建（第一阶段通路修复验证）")
    print("=" * 78)

    for dataset_path, lexicon_path, label in SOURCES:
        ds_path = _WORKSPACE / dataset_path
        lex_path = _WORKSPACE / lexicon_path
        if not ds_path.is_file() or not lex_path.is_file():
            print(f"  [SKIP] {label}: 缺少 {dataset_path} 或 {lexicon_path}")
            continue
        dataset = CounterfactualDataset.load(ds_path)
        lexicon = Lexicon.load(lex_path)
        instances, stats = build_counterfactual_instances(dataset, lexicon=lexicon)
        for instance in instances:
            # 可见图谱按 id 收集，供公平性检查核对候选是否存在
            from assemmate_ext.dataset.counterfactual import materialize

            pair = dataset.get(instance.pair_id)
            documents[instance.visible_graph_id] = materialize(pair, instance.variant_id)
        all_instances.extend(instances)
        print(f"  {label:<14} pair={len(dataset.pairs)} 实例={len(instances)} "
              f"有操作上下文={stats.n_with_operation_context} "
              f"有指令线索={stats.n_with_hints}")

    if not all_instances:
        print("[FAIL] 没有构建出任何实例", file=sys.stderr)
        return 2

    print()
    print("=" * 78)
    print("① 信息可识别性检查（第三阶段）")
    print("=" * 78)
    ident = check_identifiability(all_instances)
    print(ident.format_text())

    print()
    print("=" * 78)
    print("② 强基线输入公平性检查（交付物 5）")
    print("=" * 78)
    fairness = check_input_fairness(all_instances, documents=documents)
    print(fairness.format_text())

    print()
    print("=" * 78)
    print("实例明细（模型可见输入 vs 金标准）")
    print("=" * 78)
    for instance in all_instances:
        gold = instance.gold
        print(f"  {instance.instance_id:<34} "
              f"候选={len(instance.candidate_ids)} "
              f"操作候选={len(instance.operation_candidates)} "
              f"线索={ {k: len(v) for k, v in instance.operation_hints.items() if v} }")
        print(f"      指令: {instance.full_instruction!r}  提及: {instance.mention_text!r}")
        print(f"      gold: status={gold.status} entity={gold.entity_id} "
              f"hops={gold.requires_hops}"
              + (f" 冲突来源={gold.conflict_sources}" if gold.conflict_sources else ""))

    if args.write_observable:
        payload = [instance.observable() for instance in all_instances]
        write_json(_WORKSPACE / args.write_observable, payload)
        print(f"\n模型可见输入已落盘: {args.write_observable}（{len(payload)} 条，不含 gold）")

    write_json(_WORKSPACE / args.out, {
        "n_instances": len(all_instances),
        "identifiability": ident.to_dict(),
        "fairness": fairness.to_dict(),
        "signatures": [list(map(str, visible_signature(i))) for i in all_instances],
    })
    print(f"审计报告已写入: {args.out}")
    return 0 if (ident.ok and fairness.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
