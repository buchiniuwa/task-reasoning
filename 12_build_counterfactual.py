#!/usr/bin/env python
"""构建**关系反事实**诊断数据集（第一阶段）。

三族，每族**只隔离一类关系**，保证归因干净：

| 族 | 型号 | 唯一判据 | base → counterfactual 改了什么 |
|---|---|---|---|
| interface | CF_M1 | `compatible_with` 接口边 | 只交换两条兼容边（op1 不带 acts_on，故对象只能由接口决定） |
| operation | CF_M2 | 操作的 `acts_on`（指令按"第一步"指代工序） | 只交换两条 acts_on（两个操作其余完全相同，且无兼容边） |
| state | CF_M3 | 候选节点自身的装配状态 | **不动任何边**，只翻转两个候选的状态 |

另加一个 `underdetermined` 变体（属 interface 族）：把两条兼容边都去掉 →
两个候选都说得通 → 正确答案**不唯一**，必须输出 `ambiguous`（要求 8）。

数据纪律：全部 `synthetic`、来源 `SYN-CF-*`、不引用任何真实 BOM/SOP。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.counterfactual import (               # noqa: E402
    CounterfactualDataset, CounterfactualPair, CounterfactualVariant, EdgeSpec,
    GoldAnnotation, NodeSpec,
)
from assemmate_ext.dataset.counterfactual_validate import (      # noqa: E402
    leakage_probe, validate_counterfactual_dataset,
)
from assemmate_ext.utils.console import setup_utf8_stdout        # noqa: E402
from assemmate_ext.utils.jsonio import write_json                # noqa: E402

setup_utf8_stdout()

OUT = _WORKSPACE / "data" / "synthetic" / "counterfactual_connector.json"
OUT_LEXICON = _WORKSPACE / "data" / "lexicon" / "counterfactual_lexicon.json"

#: 候选的共享表层说法（两个候选**完全相同**，只有关系能区分）
SURFACE = "接插件主体"


def _model_entry(model_id: str, constraint_id: str) -> Dict[str, Any]:
    return {
        "model_id": model_id, "display_name": model_id, "family": "CF",
        "component_types": ["connector_body"], "constraint_ids": [constraint_id],
        "state": "in_progress",
        "evidence": {"source": "manual_annotation", "ref": f"SYN-CF-{model_id}",
                     "note": "合成反事实数据"},
        "extra": {"synthetic": True},
    }


def _relation_names(pair_edges: List[EdgeSpec]) -> List[str]:
    names = ["instance_of", "has_component", "has_step", "has_operation", "has_action",
             "acts_on", "acts_at", "achieves", "constrained_by", "requires_tool",
             "compatible_with"]
    return names


# ====================================================================== 族 1：接口
def build_interface_pair() -> CounterfactualPair:
    model = "CF_M1"
    nodes = [
        NodeSpec("cf_m1_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_m1_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_m1_op1", f"{model}_OP1", "operation", model),
        # 两个候选：同型号、同类型、内名仅差一个字符、表层说法**完全相同**
        NodeSpec("cf_m1_body_1", "CFBody1_CF_M1", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_m1_body_2", "CFBody2_CF_M1", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_m1_slot_A", f"{model}_SLOT_A", "location", model),
        NodeSpec("cf_m1_slot_B", f"{model}_SLOT_B", "location", model),
        NodeSpec("cf_m1_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_m1_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("connector_body", "connector_body", "component_type", None),
        NodeSpec("insert", "insert", "action", None),
        NodeSpec("insertion_tool", "insertion_tool", "tool", None),
    ]
    shared = [
        EdgeSpec("cf_m1_model", "has_step", "cf_m1_step1", model, "工序"),
        EdgeSpec("cf_m1_step1", "has_operation", "cf_m1_op1", model, "工序包含该操作"),
        EdgeSpec("cf_m1_model", "has_component", "cf_m1_body_1", model, "型号含该部件实例"),
        EdgeSpec("cf_m1_model", "has_component", "cf_m1_body_2", model, "型号含该部件实例"),
        EdgeSpec("cf_m1_body_1", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_m1_body_2", "instance_of", "connector_body", model, "类型归属"),
        # 注意：op1 **没有 acts_on** —— 作用对象只能由接口兼容关系决定（隔离干净）
        EdgeSpec("cf_m1_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_m1_op1", "acts_at", "cf_m1_slot_A", model, "在 A 工位槽作业"),
        EdgeSpec("cf_m1_op1", "achieves", "cf_m1_seated", model, "目标状态"),
        EdgeSpec("cf_m1_op1", "constrained_by", "cf_m1_force", model, "工艺约束"),
        EdgeSpec("cf_m1_op1", "requires_tool", "insertion_tool", model, "工具"),
    ]
    instruction = "把接插件主体插入 A 工位槽并到位"
    candidates = ["cf_m1_body_1", "cf_m1_body_2"]

    base = CounterfactualVariant(
        variant_id="base", description="body_1 与 A 工位槽兼容 → 答案是 body_1",
        add_edges=[EdgeSpec("cf_m1_body_1", "compatible_with", "cf_m1_slot_A", model,
                            "接口：body_1 可插 A 工位槽"),
                   EdgeSpec("cf_m1_body_2", "compatible_with", "cf_m1_slot_B", model,
                            "接口：body_2 可插 B 工位槽")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity="cf_m1_body_1", acceptable_entities=["cf_m1_body_1"],
            depends_on="接口兼容关系：body_1 --compatible_with--> A 工位槽",
            decisive_triples=["cf_m1_body_1|compatible_with|cf_m1_slot_A"],
            evidence_paths=[["cf_m1_body_1", "compatible_with", "cf_m1_slot_A"],
                            ["cf_m1_op1", "acts_at", "cf_m1_slot_A"]],
            candidate_ids=candidates,
            candidate_surface_names=[SURFACE, SURFACE],
            note=instruction),
    )
    counter = CounterfactualVariant(
        variant_id="counterfactual",
        description="交换两条兼容边 → 答案是 body_2（名称与候选集完全不变）",
        remove_edges=[("cf_m1_body_1", "compatible_with", "cf_m1_slot_A"),
                      ("cf_m1_body_2", "compatible_with", "cf_m1_slot_B")],
        add_edges=[EdgeSpec("cf_m1_body_2", "compatible_with", "cf_m1_slot_A", model,
                            "接口：body_2 可插 A 工位槽"),
                   EdgeSpec("cf_m1_body_1", "compatible_with", "cf_m1_slot_B", model,
                            "接口：body_1 可插 B 工位槽")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity="cf_m1_body_2", acceptable_entities=["cf_m1_body_2"],
            depends_on="接口兼容关系：body_2 --compatible_with--> A 工位槽",
            decisive_triples=["cf_m1_body_2|compatible_with|cf_m1_slot_A"],
            evidence_paths=[["cf_m1_body_2", "compatible_with", "cf_m1_slot_A"],
                            ["cf_m1_op1", "acts_at", "cf_m1_slot_A"]],
            candidate_ids=candidates,
            candidate_surface_names=[SURFACE, SURFACE],
            note=instruction),
    )
    under = CounterfactualVariant(
        variant_id="underdetermined",
        description=("两条兼容边都去掉，且操作未给出作用对象 -> "
                     "**没有任何候选被关系证据支持** -> unknown"),
        remove_edges=[("cf_m1_body_1", "compatible_with", "cf_m1_slot_A"),
                      ("cf_m1_body_2", "compatible_with", "cf_m1_slot_B")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="unknown",
            expected_entity=None, acceptable_entities=candidates,
            depends_on=("证据不足：接口关系缺失，且操作未给出作用对象 -> "
                        "无任何证据支持任一候选"),
            decisive_triples=[],
            evidence_paths=[],
            candidate_ids=candidates,
            candidate_surface_names=[SURFACE, SURFACE],
            note=instruction + "（此变体删去接口证据，正确输出应是 ambiguous）"),
    )
    return CounterfactualPair(
        pair_id="cf_interface_1", family="interface",
        depends_on="接口兼容关系 compatible_with",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_m1_force")],
        relations=_relation_names(shared),
        variants={"base": base, "counterfactual": counter, "underdetermined": under},
        meta={"instruction": instruction,
              "candidate_text_source": "surface",
              "note": "两个候选的内名仅差一个字符，表层说法完全相同"},
    )


# ====================================================================== 族 2：工序
def build_operation_pair() -> CounterfactualPair:
    model = "CF_M2"
    nodes = [
        NodeSpec("cf_m2_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_m2_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_m2_step2", f"{model}_STEP2", "step", model),
        NodeSpec("cf_m2_op1", f"{model}_OP1", "operation", model),
        NodeSpec("cf_m2_op2", f"{model}_OP2", "operation", model),
        NodeSpec("cf_m2_body_1", "CFBody1_CF_M2", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_m2_body_2", "CFBody2_CF_M2", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_m2_slot_A", f"{model}_SLOT_A", "location", model),
        NodeSpec("cf_m2_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_m2_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("connector_body", "connector_body", "component_type", None),
        NodeSpec("insert", "insert", "action", None),
    ]
    shared = [
        EdgeSpec("cf_m2_model", "has_step", "cf_m2_step1", model, "第一步"),
        EdgeSpec("cf_m2_model", "has_step", "cf_m2_step2", model, "第二步"),
        EdgeSpec("cf_m2_step1", "has_operation", "cf_m2_op1", model, "第一步的操作"),
        EdgeSpec("cf_m2_step2", "has_operation", "cf_m2_op2", model, "第二步的操作"),
        EdgeSpec("cf_m2_model", "has_component", "cf_m2_body_1", model, "部件实例"),
        EdgeSpec("cf_m2_model", "has_component", "cf_m2_body_2", model, "部件实例"),
        EdgeSpec("cf_m2_body_1", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_m2_body_2", "instance_of", "connector_body", model, "类型归属"),
        # 两个操作**除 acts_on 外完全相同**，且**没有兼容边** → 对象只能由工序决定
        EdgeSpec("cf_m2_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_m2_op1", "acts_at", "cf_m2_slot_A", model, "位置"),
        EdgeSpec("cf_m2_op1", "achieves", "cf_m2_seated", model, "目标状态"),
        EdgeSpec("cf_m2_op1", "constrained_by", "cf_m2_force", model, "约束"),
        EdgeSpec("cf_m2_op2", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_m2_op2", "acts_at", "cf_m2_slot_A", model, "位置"),
        EdgeSpec("cf_m2_op2", "achieves", "cf_m2_seated", model, "目标状态"),
        EdgeSpec("cf_m2_op2", "constrained_by", "cf_m2_force", model, "约束"),
    ]
    instruction = "第一步把接插件主体插入 A 工位槽并到位"
    candidates = ["cf_m2_body_1", "cf_m2_body_2"]

    base = CounterfactualVariant(
        variant_id="base", description="第一步的操作作用于 body_1 → 答案是 body_1",
        add_edges=[EdgeSpec("cf_m2_op1", "acts_on", "cf_m2_body_1", model,
                            "第一步操作的作用对象"),
                   EdgeSpec("cf_m2_op2", "acts_on", "cf_m2_body_2", model,
                            "第二步操作的作用对象")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity="cf_m2_body_1", acceptable_entities=["cf_m2_body_1"],
            depends_on="工序关系：第一步 --has_operation--> op1 --acts_on--> body_1",
            decisive_triples=["cf_m2_op1|acts_on|cf_m2_body_1"],
            evidence_paths=[["cf_m2_step1", "has_operation", "cf_m2_op1"],
                            ["cf_m2_op1", "acts_on", "cf_m2_body_1"]],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            note=instruction),
    )
    counter = CounterfactualVariant(
        variant_id="counterfactual",
        description="交换两条 acts_on → 答案是 body_2",
        remove_edges=[("cf_m2_op1", "acts_on", "cf_m2_body_1"),
                      ("cf_m2_op2", "acts_on", "cf_m2_body_2")],
        add_edges=[EdgeSpec("cf_m2_op1", "acts_on", "cf_m2_body_2", model,
                            "第一步操作的作用对象"),
                   EdgeSpec("cf_m2_op2", "acts_on", "cf_m2_body_1", model,
                            "第二步操作的作用对象")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity="cf_m2_body_2", acceptable_entities=["cf_m2_body_2"],
            depends_on="工序关系：第一步 --has_operation--> op1 --acts_on--> body_2",
            decisive_triples=["cf_m2_op1|acts_on|cf_m2_body_2"],
            evidence_paths=[["cf_m2_step1", "has_operation", "cf_m2_op1"],
                            ["cf_m2_op1", "acts_on", "cf_m2_body_2"]],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            note=instruction),
    )
    return CounterfactualPair(
        pair_id="cf_operation_1", family="operation",
        depends_on="工序/操作关系 acts_on（经 has_operation 由工序指代）",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_m2_force")],
        relations=_relation_names(shared),
        variants={"base": base, "counterfactual": counter},
        meta={"instruction": instruction, "candidate_text_source": "surface"},
    )


# ====================================================================== 族 3：状态
def build_state_pair() -> CounterfactualPair:
    model = "CF_M3"
    nodes = [
        NodeSpec("cf_m3_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_m3_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_m3_op1", f"{model}_OP1", "operation", model),
        NodeSpec("cf_m3_body_1", "CFBody1_CF_M3", "component_instance", model,
                 aliases=[SURFACE], assembly_state="completed"),
        NodeSpec("cf_m3_body_2", "CFBody2_CF_M3", "component_instance", model,
                 aliases=[SURFACE], assembly_state="not_started"),
        NodeSpec("cf_m3_slot_A", f"{model}_SLOT_A", "location", model),
        NodeSpec("cf_m3_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_m3_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("connector_body", "connector_body", "component_type", None),
        NodeSpec("insert", "insert", "action", None),
    ]
    shared = [
        EdgeSpec("cf_m3_model", "has_step", "cf_m3_step1", model, "工序"),
        EdgeSpec("cf_m3_step1", "has_operation", "cf_m3_op1", model, "操作"),
        EdgeSpec("cf_m3_model", "has_component", "cf_m3_body_1", model, "部件实例"),
        EdgeSpec("cf_m3_model", "has_component", "cf_m3_body_2", model, "部件实例"),
        EdgeSpec("cf_m3_body_1", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_m3_body_2", "instance_of", "connector_body", model, "类型归属"),
        # 既无 acts_on 也无 compatible_with → **唯一判据是候选自身的装配状态**
        EdgeSpec("cf_m3_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_m3_op1", "acts_at", "cf_m3_slot_A", model, "位置"),
        EdgeSpec("cf_m3_op1", "achieves", "cf_m3_seated", model, "目标状态"),
        EdgeSpec("cf_m3_op1", "constrained_by", "cf_m3_force", model, "约束"),
    ]
    instruction = "把还没装上的接插件主体装到 A 工位槽"
    candidates = ["cf_m3_body_1", "cf_m3_body_2"]

    base = CounterfactualVariant(
        variant_id="base",
        description="body_1 已完成、body_2 未开始 → 还没装的是 body_2",
        set_node_state={"cf_m3_body_1": "completed", "cf_m3_body_2": "not_started"},
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity="cf_m3_body_2", acceptable_entities=["cf_m3_body_2"],
            depends_on="节点装配状态：body_1=completed，body_2=not_started",
            decisive_triples=[],
            evidence_paths=[["cf_m3_body_2", "has_component", "cf_m3_model"]],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            note=instruction),
    )
    counter = CounterfactualVariant(
        variant_id="counterfactual",
        description="翻转两个候选的状态 → 还没装的是 body_1（边完全不动）",
        set_node_state={"cf_m3_body_1": "not_started", "cf_m3_body_2": "completed"},
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity="cf_m3_body_1", acceptable_entities=["cf_m3_body_1"],
            depends_on="节点装配状态：body_2=completed，body_1=not_started",
            decisive_triples=[],
            evidence_paths=[["cf_m3_body_1", "has_component", "cf_m3_model"]],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            note=instruction),
    )
    return CounterfactualPair(
        pair_id="cf_state_1", family="state",
        depends_on="候选节点自身的装配状态（不涉及任何边）",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_m3_force")],
        relations=_relation_names(shared),
        variants={"base": base, "counterfactual": counter},
        meta={"instruction": instruction, "candidate_text_source": "surface"},
    )


def build_lexicon() -> Dict[str, Any]:
    """反事实数据集专用词表。

    **必需**：`resolve_instruction_hints` 靠词表说法把"第一步""A 工位槽"匹配到节点；
    没有它就只能匹配节点内名（如 CF_M1_SLOT_A），而指令里根本不会出现内名。
    """
    entries = [
        {"node_id": "cf_m1_body_1", "zh": [SURFACE]},
        {"node_id": "cf_m1_body_2", "zh": [SURFACE]},
        {"node_id": "cf_m2_body_1", "zh": [SURFACE]},
        {"node_id": "cf_m2_body_2", "zh": [SURFACE]},
        {"node_id": "cf_m3_body_1", "zh": [SURFACE]},
        {"node_id": "cf_m3_body_2", "zh": [SURFACE]},
        {"node_id": "cf_m1_slot_A", "zh": ["A 工位槽"]},
        {"node_id": "cf_m1_slot_B", "zh": ["B 工位槽"]},
        {"node_id": "cf_m2_slot_A", "zh": ["A 工位槽"]},
        {"node_id": "cf_m3_slot_A", "zh": ["A 工位槽"]},
        {"node_id": "cf_m1_step1", "zh": ["第一步"]},
        {"node_id": "cf_m2_step1", "zh": ["第一步"]},
        {"node_id": "cf_m2_step2", "zh": ["第二步"]},
        {"node_id": "cf_m3_step1", "zh": ["第一步"]},
        {"node_id": "insert", "zh": ["插入"]},
        {"node_id": "cf_m1_seated", "zh": ["到位"]},
        {"node_id": "cf_m2_seated", "zh": ["到位"]},
        {"node_id": "cf_m3_seated", "zh": ["到位"]},
        {"node_id": "cf_m1_force", "zh": ["压紧力按 15–20N"]},
        {"node_id": "cf_m2_force", "zh": ["压紧力按 15–20N"]},
        {"node_id": "cf_m3_force", "zh": ["压紧力按 15–20N"]},
    ]
    payload = {"meta": {"synthetic": True,
                        "purpose": "反事实数据集专用词表：让指令线索能被确定性匹配到节点",
                        "note": "合成数据"},
               "entries": entries}
    write_json(OUT_LEXICON, payload)
    return payload


def run_gap_audit(dataset) -> Dict[str, Any]:
    """逐变体跑三条路径并**分开报告**：生产(词表线索) / oracle(诊断上限) / 规则查询(B2)。"""
    from assemmate_ext.dataset.builders import Lexicon
    from assemmate_ext.dataset.counterfactual import materialize
    from assemmate_ext.grounding.operation_context import (
        audit_information_gap, build_operation_context,
    )

    lexicon = Lexicon.load(OUT_LEXICON)
    oracle_ops = {"interface": "cf_m1_op1", "operation": "cf_m2_op1", "state": "cf_m3_op1"}
    rows: List[Dict[str, Any]] = []
    for pair in dataset.pairs:
        instruction = pair.meta.get("instruction", "")
        model_id = pair.models[0]["model_id"] if pair.models else None
        for variant_id, variant in pair.variants.items():
            document = materialize(pair, variant_id)
            gold = variant.gold
            row: Dict[str, Any] = {
                "pair_id": pair.pair_id, "variant": variant_id, "family": pair.family,
                "instruction": instruction,
                "expected": {"status": gold.expected_status, "entity": gold.expected_entity},
            }
            hints_ctx = build_operation_context(
                document, gold.mention, gold.slot, gold.candidate_ids,
                instruction=instruction, model_id=model_id, lexicon=lexicon)
            row["hints_path"] = audit_information_gap(
                document, gold.expected_entity, gold.candidate_ids, hints_ctx,
                expected_status=gold.expected_status)
            oracle_ctx = build_operation_context(
                document, gold.mention, gold.slot, gold.candidate_ids,
                instruction=instruction, model_id=model_id, lexicon=lexicon,
                oracle_op_id=oracle_ops.get(pair.family))
            row["oracle_path"] = audit_information_gap(
                document, gold.expected_entity, gold.candidate_ids, oracle_ctx,
                expected_status=gold.expected_status)
            rows.append(row)
    return {"pairs": rows,
            "note": ("hints_path=生产可用（词表线索）；oracle_path=诊断上限（用 gold op_id）。"
                     "两者必须分开报告。")}


def main() -> int:
    dataset = CounterfactualDataset(
        name="counterfactual_connector",
        pairs=[build_interface_pair(), build_operation_pair(), build_state_pair()],
        meta={
            "synthetic": True,
            "purpose": ("关系反事实诊断集：实体名称/类型/型号都无法区分候选，"
                        "答案只能由接口、工序或状态关系决定"),
            "candidate_text_source": "surface（评测时必须用共享表层说法作为候选文本，"
                                     "不能用内部节点名，否则引入非关系线索）",
            "leakage_control": ("同一对的两个变体共享完全相同的节点表（含顺序）与指令，"
                                "金标准在候选表中的索引不同 → 『固定选第 N 个』只能对一半"),
            "note": "合成数据；不得作为真实产品资料引用",
        },
    )
    dataset.save(OUT)

    report = validate_counterfactual_dataset(dataset)
    probe = leakage_probe(dataset)

    print("=" * 78)
    print("关系反事实数据集（第一阶段）")
    print("=" * 78)
    for pair in dataset.pairs:
        n_cand = len(pair.variants["base"].gold.candidate_ids)
        print(f"  {pair.pair_id:<16} 族={pair.family:<10} 变体={list(pair.variants)} "
              f"候选={n_cand}")
    print()
    print(report.format_text())
    print()
    print("泄漏捷径量化（成对正确率；越接近 0 越好）：")
    for key, value in probe["shortcut_pairwise_accuracy"].items():
        print(f"    {key:<16} {value}")
    print(f"    （探针 pair 数 = {probe['n_pairs_probed']}）")
    print()
    for row in probe["pairs"]:
        print(f"    {row['pair_id']:<16} 候选={row['candidates']} "
              f"gold={row['gold']} 捷径={row['shortcut_pairwise_correct']}")
    print(f"\n已写入: {OUT}")

    lex = build_lexicon()
    print()
    print(f"  词表已写入: {OUT_LEXICON}（{len(lex['entries'])} 条）")

    audit = run_gap_audit(dataset)
    print()
    print("=" * 78)
    print("信息缺口审计（生产路径 = 词表线索；oracle 路径 = 诊断上限）")
    print("=" * 78)
    for row in audit["pairs"]:
        exp = row["expected"]
        hp = row["hints_path"]["rule_query"]
        op = row["oracle_path"]["rule_query"]
        mark = "OK " if row["hints_path"]["rule_matches_gold"] else "!! "
        print(f"  {mark}{row['pair_id']:<15} {row['variant']:<15} "
              f"期望={exp['status']}/{exp['entity']}")
        print(f"      生产(线索)={hp['status']}/{hp['entity']}")
        for item in hp["filters_applied"]:
            print(f"         过滤: {item}")
        print(f"      oracle     ={op['status']}/{op['entity']}"
              f"  source={row['oracle_path']['context_source']}"
              f" 生产安全={row['oracle_path']['is_production_safe']}")
        for item in row["hints_path"]["missing_signals"]:
            print(f"      缺信号: {item}")

    write_json(_WORKSPACE / "outputs" / "counterfactual_validation.json",
               {"validation": report.to_dict(), "leakage_probe": probe, "gap_audit": audit})
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
