#!/usr/bin/env python
"""分口径覆盖率报告（第一阶段 问题一）。

把上一轮那句会被误读的"规则覆盖率 10/15"拆成**各自独立**的指标：

    · 唯一实体解析覆盖率（gold=grounded 上答对的比例）
    · grounded 实体准确率（precision）
    · ambiguous / unknown / conflict 状态判断准确率
    · conflict 检测 P/R/F1
    · 选择性预测覆盖率与覆盖精度
    · **正确弃权率**（不该给唯一答案时正确弃权的比例）

并同时给出 B2 / B2+（仅冲突识别）/ B2+（冲突识别 + 可信度择优）三条确定性基线，
用来回答"10/15 到底是规则不行，还是规则没拿到它需要的输入通道"。

用法：
    python scripts/15_coverage_report.py [--out outputs/coverage_report.json]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.builders import Lexicon                      # noqa: E402
from assemmate_ext.dataset.counterfactual import (                      # noqa: E402
    CounterfactualDataset, materialize,
)
from assemmate_ext.grounding.baselines import (                         # noqa: E402
    detect_source_conflict, deterministic_relational_query_with_trust, source_trust_table,
)
from assemmate_ext.grounding.dataset import (                           # noqa: E402
    GroundingInstance, build_counterfactual_instances,
)
from assemmate_ext.grounding.metrics import (                           # noqa: E402
    Prediction, evaluate, format_report, prediction_from_query_result,
)
from assemmate_ext.grounding.operation_context import (                 # noqa: E402
    InstructionHints, build_operation_context, deterministic_relational_query,
)
from assemmate_ext.utils.console import setup_utf8_stdout               # noqa: E402
from assemmate_ext.utils.jsonio import write_json                       # noqa: E402

setup_utf8_stdout()

SOURCES = [
    ("data/synthetic/counterfactual_connector.json",
     "data/lexicon/counterfactual_lexicon.json", "完整证据三族"),
    ("data/synthetic/counterfactual_partial.json",
     "data/lexicon/counterfactual_partial_lexicon.json", "部分可观测四族"),
]


# ====================================================================== 载入
def load_instances():
    instances: List[GroundingInstance] = []
    documents: Dict[str, Any] = {}
    for dataset_path, lexicon_path, _label in SOURCES:
        ds_path, lex_path = _WORKSPACE / dataset_path, _WORKSPACE / lexicon_path
        if not ds_path.is_file() or not lex_path.is_file():
            continue
        dataset = CounterfactualDataset.load(ds_path)
        lexicon = Lexicon.load(lex_path)
        batch, _stats = build_counterfactual_instances(dataset, lexicon=lexicon)
        for instance in batch:
            pair = dataset.get(instance.pair_id)
            documents[instance.visible_graph_id] = materialize(pair, instance.variant_id)
        instances.extend(batch)
    return instances, documents


def _context(document: Any, instance: GroundingInstance):
    """用**实例已经观测到的**线索重建上下文（与 audit._rule_solvable 同一份输入）。"""
    hints_data = dict(instance.operation_hints or {})
    hints = InstructionHints(
        instruction=str(hints_data.get("instruction") or instance.full_instruction),
        mention=str(hints_data.get("mention") or instance.mention_text),
        location_ids=list(hints_data.get("location_ids") or []),
        step_ids=list(hints_data.get("step_ids") or []),
        state_ids=list(hints_data.get("state_ids") or []),
        action_ids=list(hints_data.get("action_ids") or []),
        tool_ids=list(hints_data.get("tool_ids") or []),
        hints_source=str(hints_data.get("hints_source") or "lexicon_match"))
    return build_operation_context(
        document, instance.mention_text, instance.slot, instance.candidate_ids,
        instruction=instance.full_instruction, model_id=instance.model_id,
        completed_ops=instance.completed_ops, current_state=instance.current_state,
        hints=hints, operation_candidates=instance.operation_candidates)


# ====================================================================== 三个基线
def run_b2(instances, documents) -> List[Prediction]:
    out = []
    for instance in instances:
        doc = documents.get(instance.visible_graph_id)
        result = deterministic_relational_query(doc, instance.candidate_ids,
                                                _context(doc, instance))
        out.append(prediction_from_query_result(instance, result, system="B2"))
    return out


def run_b2plus(instances, documents, use_trust_ordering: bool) -> List[Prediction]:
    name = "B2+" if use_trust_ordering else "B2+ (仅冲突)"
    out = []
    for instance in instances:
        doc = documents.get(instance.visible_graph_id)
        result = deterministic_relational_query_with_trust(
            doc, instance.candidate_ids, _context(doc, instance),
            use_trust_ordering=use_trust_ordering)
        out.append(prediction_from_query_result(instance, result, system=name))
    return out


def by_family(instances, predictions) -> Dict[str, Dict[str, Any]]:
    """按族分别评测（把 P3 单独拎出来看，避免被平均掉）。"""
    buckets: Dict[str, List[GroundingInstance]] = defaultdict(list)
    for instance in instances:
        family = (instance.scenario or "").replace("counterfactual_", "") or "unknown"
        buckets[family].append(instance)
    ids = {p.instance_id: p for p in predictions}
    return {family: evaluate(items, [ids[i.instance_id] for i in items], system=family)
            for family, items in sorted(buckets.items())}


# ====================================================================== 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="分口径覆盖率报告")
    parser.add_argument("--out", default="outputs/coverage_report.json")
    args = parser.parse_args()

    instances, documents = load_instances()
    if not instances:
        print("[FAIL] 没有构建出任何实例", file=sys.stderr)
        return 2

    systems = {
        "B2": run_b2(instances, documents),
        "B2+ (仅冲突)": run_b2plus(instances, documents, use_trust_ordering=False),
        "B2+": run_b2plus(instances, documents, use_trust_ordering=True),
    }
    results = {name: evaluate(instances, preds, system=name)
               for name, preds in systems.items()}

    print("=" * 100)
    print(f"分口径覆盖率报告（{len(instances)} 条实例）")
    print("=" * 100)
    print(format_report(list(results.values())))

    print()
    print("=" * 100)
    print("gold 状态分布")
    print("=" * 100)
    print(f"  {results['B2']['status_counts_gold']}")

    print()
    print("=" * 100)
    print("逐状态判断准确率 / conflict 检测")
    print("=" * 100)
    for name, res in results.items():
        parts = []
        for status, cell in res["status_accuracy"].items():
            if cell["n"]:
                parts.append(f"{status}={cell['correct']}/{cell['n']}")
        cd = res["conflict_detection"]
        print(f"  {name:<14} " + "  ".join(parts)
              + f"   | conflict P={cd['precision']} R={cd['recall']} F1={cd['f1']}")

    print()
    print("=" * 100)
    print("gold 状态 -> 系统状态 混淆")
    print("=" * 100)
    for name, res in results.items():
        print(f"  {name}")
        for gold_status, row in res["confusion_gold_to_pred"].items():
            print(f"      {gold_status:<10} -> {row}")

    print()
    print("=" * 100)
    print("按族分解（唯一解析覆盖率 / 正确弃权率）")
    print("=" * 100)
    family_tables = {name: by_family(instances, preds) for name, preds in systems.items()}
    families = sorted({f for table in family_tables.values() for f in table})
    print(f"  {'族':<24}" + "".join(f"{name:<20}" for name in systems))
    for family in families:
        cells = []
        for name in systems:
            cell = family_tables[name].get(family)
            if not cell:
                cells.append("-")
                continue
            ur, ab = cell["unique_resolution"], cell["abstention"]
            pieces = []
            if ur["n_gold_grounded"]:
                pieces.append(f"解析 {ur['correct']}/{ur['n_gold_grounded']}")
            if ab["n_gold_nonunique"]:
                pieces.append(f"弃权 {ab['correct_abstention']}/{ab['n_gold_nonunique']}")
            cells.append(" ".join(pieces) or "-")
        print(f"  {family:<24}" + "".join(f"{c:<20}" for c in cells))

    print()
    print("=" * 100)
    print("P3（来源冲突）单独看：B2 的 ambiguous 到底算不算失败")
    print("=" * 100)
    for instance in instances:
        if "conflict" not in (instance.scenario or ""):
            continue
        doc = documents.get(instance.visible_graph_id)
        ctx = _context(doc, instance)
        conflict = detect_source_conflict(doc, instance.candidate_ids, ctx)
        print(f"  {instance.instance_id}")
        print(f"      gold={instance.gold.status}  冲突来源={instance.gold.conflict_sources}")
        if conflict:
            print(f"      检测到的冲突: 操作={conflict.operation_id} 关系={conflict.relation}")
            print(f"      各来源断言={conflict.assertions}")
            print(f"      来源可信度（**先验假设**）={conflict.trust}")
            print(f"      按可信度择优（次要答案）={conflict.higher_trust_entity}"
                  f"  （gold 的 higher_trust_entity={instance.gold.higher_trust_entity}）")
            print(f"      参与冲突的三元组（事实证据）={conflict.triples}")

    print()
    print("来源可信度先验表（**合成实验假设，非工业数据标定**）：")
    for source, value in sorted(source_trust_table().items(), key=lambda kv: -kv[1]):
        print(f"    {source:<22} {value}")

    write_json(_WORKSPACE / args.out, {
        "n_instances": len(instances),
        "overall": results,
        "by_family": family_tables,
        "source_trust_prior": source_trust_table(),
        "disclaimer": ("SOURCE_TRUST_PRIOR 与合成数据里的 0.9/0.35 均为**实验假设**，"
                       "不是经工业数据标定的真实置信度"),
        "predictions": {name: [p.to_dict() for p in preds]
                        for name, preds in systems.items()},
    })
    print(f"\n报告已写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
