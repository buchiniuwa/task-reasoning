#!/usr/bin/env python
"""构建**部分可观测**反事实数据集（第二阶段 P1–P4）。

与已有的三族（完整证据反事实）并列，这四族专门制造"证据不完整"的情形，
并要求模型/基线**如实区分**四种结论：

| 族 | 情形 | 正确结论 | 为什么不能靠名字/顺序 |
|---|---|---|---|
| **P1** `indirect_inference` | 直接接口边缺失，但**类型级**兼容可间接推断（2 跳） | grounded（唯一） | 候选同名同类型，直连边不存在 |
| **P2** `multihop_operation` | 指令只说"第一步"，位置需经 工序→操作→位置 推断（3 跳） | grounded（唯一） | 指令里**没有位置词**（`instruction_omits` 机器检查） |
| **P3** `source_conflict` | 同一条关系由两个来源给出互相矛盾的结论 | **conflict** | 两个候选都有证据，冲突在来源层面 |
| **P4** `insufficient_evidence` | ①两个并行操作各自支持一个候选 ②证据被移除 | ①**ambiguous** ②**unknown** | 命名与顺序都无法区分两者 |

纪律：全部 `synthetic`、来源 `SYN-PO-*`；每个样本保留可见 KG 与 gold（gold 只给评测器）；
**不假设"被遮蔽边对应的原实体就是唯一答案"** —— 每个变体的 gold 都按可见信息重新判定。
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

OUT = _WORKSPACE / "data" / "synthetic" / "counterfactual_partial.json"
OUT_LEXICON = _WORKSPACE / "data" / "lexicon" / "counterfactual_partial_lexicon.json"
SURFACE = "接插件主体"

RELATIONS = ["instance_of", "has_component", "has_step", "has_operation", "has_action",
             "acts_on", "acts_at", "achieves", "constrained_by", "compatible_with"]


def _model_entry(model_id: str, constraint_id: str,
                 component_types=("connector_body",)) -> Dict[str, Any]:
    return {
        "model_id": model_id, "display_name": model_id, "family": "CF-PO",
        "component_types": list(component_types), "constraint_ids": [constraint_id],
        "state": "in_progress",
        "evidence": {"source": "manual_annotation", "ref": f"SYN-PO-{model_id}",
                     "note": "合成部分可观测数据"},
        "extra": {"synthetic": True},
    }


# ====================================================================== P1 间接推断
def build_p1() -> CounterfactualPair:
    model = "CF_P1"
    nodes = [
        NodeSpec("cf_p1_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_p1_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_p1_op1", f"{model}_OP1", "operation", model),
        NodeSpec("cf_p1_body_1", "CFBody1_CF_P1", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p1_body_2", "CFBody2_CF_P1", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p1_type_8pin", "connector_8pin", "component_type", None),
        NodeSpec("cf_p1_type_12pin", "connector_12pin", "component_type", None),
        NodeSpec("cf_p1_slot_A", f"{model}_SLOT_8PIN", "location", model),
        NodeSpec("cf_p1_slot_B", f"{model}_SLOT_12PIN", "location", model),
        NodeSpec("cf_p1_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_p1_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("insert", "insert", "action", None),
    ]
    shared = [
        EdgeSpec("cf_p1_model", "has_step", "cf_p1_step1", model, "工序"),
        EdgeSpec("cf_p1_step1", "has_operation", "cf_p1_op1", model, "操作"),
        EdgeSpec("cf_p1_model", "has_component", "cf_p1_body_1", model, "部件实例"),
        EdgeSpec("cf_p1_model", "has_component", "cf_p1_body_2", model, "部件实例"),
        # 候选只挂到**类型**上；类型级别才有接口信息 -> 必须 2 跳推断
        EdgeSpec("cf_p1_body_1", "instance_of", "cf_p1_type_8pin", model, "属于八针类型"),
        EdgeSpec("cf_p1_body_2", "instance_of", "cf_p1_type_12pin", model, "属于十二针类型"),
        EdgeSpec("cf_p1_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_p1_op1", "acts_at", "cf_p1_slot_A", model, "在八针工位作业"),
        EdgeSpec("cf_p1_op1", "achieves", "cf_p1_seated", model, "目标状态"),
        EdgeSpec("cf_p1_op1", "constrained_by", "cf_p1_force", model, "约束"),
    ]
    instruction = "把接插件主体插到八针工位并到位"
    candidates = ["cf_p1_body_1", "cf_p1_body_2"]

    def variant(variant_id: str, description: str, gold_body: str,
                removed: List[tuple], added: List[EdgeSpec]) -> CounterfactualVariant:
        # 金标准候选所属的**类型**（P1 的关键：接口信息挂在类型上）
        type_id = "cf_p1_type_8pin" if gold_body.endswith("_1") else "cf_p1_type_12pin"
        chain = [f"{gold_body}|instance_of|{type_id}",
                 f"{type_id}|compatible_with|cf_p1_slot_A"]
        return CounterfactualVariant(
            variant_id=variant_id, description=description,
            remove_edges=removed, add_edges=added,
            gold=GoldAnnotation(
                mention=SURFACE, slot="object", expected_status="grounded",
                expected_entity=gold_body, acceptable_entities=[gold_body],
                depends_on="类型级接口兼容（相对候选不可直连，需 2 跳推断）",
                decisive_triples=list(chain),
                evidence_paths=[[gold_body, "instance_of", type_id],
                                [type_id, "compatible_with", "cf_p1_slot_A"]],
                candidate_ids=candidates,
                candidate_surface_names=[SURFACE, SURFACE],
                requires_hops=2,
                inference_chain=list(chain),
                note=instruction),
        )

    base = variant(
        "base", "八针类型兼容八针工位 → 答案是 body_1（直连边不存在）",
        "cf_p1_body_1", [],
        [EdgeSpec("cf_p1_type_8pin", "compatible_with", "cf_p1_slot_A", model,
                  "类型级接口：八针可插八针工位", source="sop_document", ref="SYN-PO-SOP-P1"),
         EdgeSpec("cf_p1_type_12pin", "compatible_with", "cf_p1_slot_B", model,
                  "类型级接口：十二针可插十二针工位", source="sop_document",
                  ref="SYN-PO-SOP-P1")])
    counter = variant(
        "counterfactual", "交换类型级接口 → 答案是 body_2",
        "cf_p1_body_2",
        [("cf_p1_type_8pin", "compatible_with", "cf_p1_slot_A"),
         ("cf_p1_type_12pin", "compatible_with", "cf_p1_slot_B")],
        [EdgeSpec("cf_p1_type_12pin", "compatible_with", "cf_p1_slot_A", model,
                  "类型级接口：十二针可插八针工位", source="sop_document",
                  ref="SYN-PO-SOP-P1"),
         EdgeSpec("cf_p1_type_8pin", "compatible_with", "cf_p1_slot_B", model,
                  "类型级接口：八针可插十二针工位", source="sop_document",
                  ref="SYN-PO-SOP-P1")])
    return CounterfactualPair(
        pair_id="cf_partial_indirect_1", family="indirect_inference",
        depends_on="类型级接口兼容（需 2 跳：实例→类型→工位）",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_p1_force",
                             component_types=("connector_8pin", "connector_12pin"))],
        relations=RELATIONS,
        variants={"base": base, "counterfactual": counter},
        meta={"instruction": instruction, "candidate_text_source": "surface"})


# ====================================================================== P2 多跳工序
def build_p2() -> CounterfactualPair:
    model = "CF_P2"
    nodes = [
        NodeSpec("cf_p2_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_p2_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_p2_step2", f"{model}_STEP2", "step", model),
        NodeSpec("cf_p2_op1", f"{model}_OP1", "operation", model),
        NodeSpec("cf_p2_op2", f"{model}_OP2", "operation", model),
        NodeSpec("cf_p2_body_1", "CFBody1_CF_P2", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p2_body_2", "CFBody2_CF_P2", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p2_slot_A", f"{model}_SLOT_A", "location", model),
        NodeSpec("cf_p2_slot_B", f"{model}_SLOT_B", "location", model),
        NodeSpec("cf_p2_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_p2_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("connector_body", "connector_body", "component_type", None),
        NodeSpec("insert", "insert", "action", None),
    ]
    shared = [
        EdgeSpec("cf_p2_model", "has_step", "cf_p2_step1", model, "第一步"),
        EdgeSpec("cf_p2_model", "has_step", "cf_p2_step2", model, "第二步"),
        EdgeSpec("cf_p2_step1", "has_operation", "cf_p2_op1", model, "第一步的操作"),
        EdgeSpec("cf_p2_step2", "has_operation", "cf_p2_op2", model, "第二步的操作"),
        EdgeSpec("cf_p2_model", "has_component", "cf_p2_body_1", model, "部件实例"),
        EdgeSpec("cf_p2_model", "has_component", "cf_p2_body_2", model, "部件实例"),
        EdgeSpec("cf_p2_body_1", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_p2_body_2", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_p2_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_p2_op1", "acts_at", "cf_p2_slot_A", model, "第一步在 A 工位"),
        EdgeSpec("cf_p2_op1", "achieves", "cf_p2_seated", model, "目标状态"),
        EdgeSpec("cf_p2_op1", "constrained_by", "cf_p2_force", model, "约束"),
        EdgeSpec("cf_p2_op2", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_p2_op2", "acts_at", "cf_p2_slot_B", model, "第二步在 B 工位"),
        EdgeSpec("cf_p2_op2", "achieves", "cf_p2_seated", model, "目标状态"),
        EdgeSpec("cf_p2_op2", "constrained_by", "cf_p2_force", model, "约束"),
    ]
    instruction = "第一步把接插件主体插到位"          # ← 刻意不提位置
    omitted = ["A 工位槽", "B 工位槽"]
    candidates = ["cf_p2_body_1", "cf_p2_body_2"]

    def gold(body: str, slot: str) -> GoldAnnotation:
        return GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="grounded",
            expected_entity=body, acceptable_entities=[body],
            depends_on=("多跳工序推断：第一步 -> 操作 -> 位置 -> 接口 -> 候选"
                        "（指令未给位置）"),
            decisive_triples=["cf_p2_step1|has_operation|cf_p2_op1",
                              f"cf_p2_op1|acts_at|{slot}",
                              f"{body}|compatible_with|{slot}"],
            evidence_paths=[["cf_p2_step1", "has_operation", "cf_p2_op1"],
                            ["cf_p2_op1", "acts_at", slot],
                            [body, "compatible_with", slot]],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            requires_hops=3,
            inference_chain=["cf_p2_step1|has_operation|cf_p2_op1",
                             f"cf_p2_op1|acts_at|{slot}",
                             f"{body}|compatible_with|{slot}"],
            instruction_omits=list(omitted),
            note=instruction)

    base = CounterfactualVariant(
        variant_id="base", description="body_1 与第一步的工位(A)兼容 → 答案是 body_1",
        add_edges=[EdgeSpec("cf_p2_body_1", "compatible_with", "cf_p2_slot_A", model,
                            "接口：body_1 可插 A 工位槽", source="sop_document",
                            ref="SYN-PO-SOP-P2"),
                   EdgeSpec("cf_p2_body_2", "compatible_with", "cf_p2_slot_B", model,
                            "接口：body_2 可插 B 工位槽", source="sop_document",
                            ref="SYN-PO-SOP-P2")],
        gold=gold("cf_p2_body_1", "cf_p2_slot_A"))
    counter = CounterfactualVariant(
        variant_id="counterfactual", description="交换接口 → 答案是 body_2",
        remove_edges=[("cf_p2_body_1", "compatible_with", "cf_p2_slot_A"),
                      ("cf_p2_body_2", "compatible_with", "cf_p2_slot_B")],
        add_edges=[EdgeSpec("cf_p2_body_2", "compatible_with", "cf_p2_slot_A", model,
                            "接口：body_2 可插 A 工位槽", source="sop_document",
                            ref="SYN-PO-SOP-P2"),
                   EdgeSpec("cf_p2_body_1", "compatible_with", "cf_p2_slot_B", model,
                            "接口：body_1 可插 B 工位槽", source="sop_document",
                            ref="SYN-PO-SOP-P2")],
        gold=gold("cf_p2_body_2", "cf_p2_slot_A"))
    return CounterfactualPair(
        pair_id="cf_partial_multihop_1", family="multihop_operation",
        depends_on="工序->操作->位置->接口 的多跳链（指令未给位置）",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_p2_force")], relations=RELATIONS,
        variants={"base": base, "counterfactual": counter},
        meta={"instruction": instruction, "candidate_text_source": "surface"})


# ====================================================================== P3 来源冲突
def build_p3() -> CounterfactualPair:
    model = "CF_P3"
    nodes = [
        NodeSpec("cf_p3_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_p3_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_p3_op1", f"{model}_OP1", "operation", model),
        NodeSpec("cf_p3_body_1", "CFBody1_CF_P3", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p3_body_2", "CFBody2_CF_P3", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p3_slot_A", f"{model}_SLOT_A", "location", model),
        NodeSpec("cf_p3_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_p3_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("connector_body", "connector_body", "component_type", None),
        NodeSpec("insert", "insert", "action", None),
    ]
    shared = [
        EdgeSpec("cf_p3_model", "has_step", "cf_p3_step1", model, "工序"),
        EdgeSpec("cf_p3_step1", "has_operation", "cf_p3_op1", model, "操作"),
        EdgeSpec("cf_p3_model", "has_component", "cf_p3_body_1", model, "部件实例"),
        EdgeSpec("cf_p3_model", "has_component", "cf_p3_body_2", model, "部件实例"),
        EdgeSpec("cf_p3_body_1", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_p3_body_2", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_p3_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_p3_op1", "acts_at", "cf_p3_slot_A", model, "位置"),
        EdgeSpec("cf_p3_op1", "achieves", "cf_p3_seated", model, "目标状态"),
        EdgeSpec("cf_p3_op1", "constrained_by", "cf_p3_force", model, "约束"),
    ]
    instruction = "把接插件主体插到 A 工位槽并到位"
    candidates = ["cf_p3_body_1", "cf_p3_body_2"]
    high_ref, low_ref = "SYN-PO-SOP-P3", "SYN-PO-PRED-P3"

    def make(variant_id: str, description: str, high_body: str, low_body: str,
             removed: List[tuple]) -> CounterfactualVariant:
        return CounterfactualVariant(
            variant_id=variant_id, description=description, remove_edges=removed,
            add_edges=[
                EdgeSpec("cf_p3_op1", "acts_on", high_body, model,
                         "工艺文件说作用对象是它", source="sop_document",
                         confidence=0.9, ref=high_ref),
                EdgeSpec("cf_p3_op1", "acts_on", low_body, model,
                         "模型预测说作用对象是它（与工艺文件矛盾）",
                         source="model_prediction", confidence=0.35, ref=low_ref)],
            gold=GoldAnnotation(
                mention=SURFACE, slot="object", expected_status="conflict",
                expected_entity=None, acceptable_entities=candidates,
                depends_on="来源冲突：工艺文件与模型预测给出不同对象",
                decisive_triples=[f"cf_p3_op1|acts_on|{high_body}",
                                  f"cf_p3_op1|acts_on|{low_body}"],
                evidence_paths=[[high_body, "compatible_with", "cf_p3_slot_A"]],
                candidate_ids=candidates,
                candidate_surface_names=[SURFACE, SURFACE],
                conflict_sources=[high_ref, low_ref],
                higher_trust_entity=high_body,
                note=(instruction + "（两个来源矛盾，正确输出应是 conflict；"
                      "若按来源可信度择优，则应选高可信来源的那个）")))

    base = make("base", "工艺文件(0.9)指向 body_1，模型预测(0.35)指向 body_2",
                "cf_p3_body_1", "cf_p3_body_2", [])
    counter = make("counterfactual", "交换来源可信度 → 高可信来源改为 body_2",
                   "cf_p3_body_2", "cf_p3_body_1",
                   [("cf_p3_op1", "acts_on", "cf_p3_body_1"),
                    ("cf_p3_op1", "acts_on", "cf_p3_body_2")])
    return CounterfactualPair(
        pair_id="cf_partial_conflict_1", family="source_conflict",
        depends_on="多来源冲突（sop_document 0.9 vs model_prediction 0.35）",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_p3_force")], relations=RELATIONS,
        variants={"base": base, "counterfactual": counter},
        meta={"instruction": instruction, "candidate_text_source": "surface"})


# ====================================================================== P4 证据不足
def build_p4() -> CounterfactualPair:
    model = "CF_P4"
    nodes = [
        NodeSpec("cf_p4_model", f"{model}_MODEL", "model", model),
        NodeSpec("cf_p4_step1", f"{model}_STEP1", "step", model),
        NodeSpec("cf_p4_op1", f"{model}_OP1", "operation", model),
        NodeSpec("cf_p4_op2", f"{model}_OP2", "operation", model),
        NodeSpec("cf_p4_body_1", "CFBody1_CF_P4", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p4_body_2", "CFBody2_CF_P4", "component_instance", model,
                 aliases=[SURFACE]),
        NodeSpec("cf_p4_slot_A", f"{model}_SLOT_A", "location", model),
        NodeSpec("cf_p4_seated", f"{model}_SEATED", "target_state", model),
        NodeSpec("cf_p4_force", f"{model}_FORCE", "process_constraint", model),
        NodeSpec("connector_body", "connector_body", "component_type", None),
        NodeSpec("insert", "insert", "action", None),
    ]
    # 两个操作**完全对称**（动作/位置/状态/约束都一样），唯一差别是作用对象
    shared = [
        EdgeSpec("cf_p4_model", "has_step", "cf_p4_step1", model, "工序"),
        EdgeSpec("cf_p4_step1", "has_operation", "cf_p4_op1", model, "操作一"),
        EdgeSpec("cf_p4_step1", "has_operation", "cf_p4_op2", model, "操作二"),
        EdgeSpec("cf_p4_model", "has_component", "cf_p4_body_1", model, "部件实例"),
        EdgeSpec("cf_p4_model", "has_component", "cf_p4_body_2", model, "部件实例"),
        EdgeSpec("cf_p4_body_1", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_p4_body_2", "instance_of", "connector_body", model, "类型归属"),
        EdgeSpec("cf_p4_op1", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_p4_op1", "acts_at", "cf_p4_slot_A", model, "位置"),
        EdgeSpec("cf_p4_op1", "achieves", "cf_p4_seated", model, "目标状态"),
        EdgeSpec("cf_p4_op1", "constrained_by", "cf_p4_force", model, "约束"),
        EdgeSpec("cf_p4_op2", "has_action", "insert", model, "动作"),
        EdgeSpec("cf_p4_op2", "acts_at", "cf_p4_slot_A", model, "位置"),
        EdgeSpec("cf_p4_op2", "achieves", "cf_p4_seated", model, "目标状态"),
        EdgeSpec("cf_p4_op2", "constrained_by", "cf_p4_force", model, "约束"),
    ]
    instruction = "把接插件主体插到 A 工位槽并到位"
    candidates = ["cf_p4_body_1", "cf_p4_body_2"]

    ambiguous = CounterfactualVariant(
        variant_id="base",
        description="两个对称操作各支持一个候选 → 两个读法都成立 → ambiguous",
        add_edges=[EdgeSpec("cf_p4_op1", "acts_on", "cf_p4_body_1", model,
                            "操作一作用于它", source="sop_document", ref="SYN-PO-SOP-P4"),
                   EdgeSpec("cf_p4_op2", "acts_on", "cf_p4_body_2", model,
                            "操作二作用于它", source="sop_document", ref="SYN-PO-SOP-P4")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="ambiguous",
            expected_entity=None, acceptable_entities=candidates,
            depends_on="两个对称操作的证据无法区分两个候选",
            decisive_triples=["cf_p4_op1|acts_on|cf_p4_body_1",
                              "cf_p4_op2|acts_on|cf_p4_body_2"],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            note=instruction + "（两个候选各有独立证据、无法区分 -> ambiguous）"))
    unknown = CounterfactualVariant(
        variant_id="counterfactual",
        description="把两个 acts_on 都去掉 → 没有任何证据支持任一候选 → unknown",
        remove_edges=[("cf_p4_op1", "acts_on", "cf_p4_body_1"),
                      ("cf_p4_op2", "acts_on", "cf_p4_body_2")],
        gold=GoldAnnotation(
            mention=SURFACE, slot="object", expected_status="unknown",
            expected_entity=None, acceptable_entities=candidates,
            depends_on="操作未给出作用对象，且无接口关系 -> 缺乏必要证据",
            decisive_triples=[],
            candidate_ids=candidates, candidate_surface_names=[SURFACE, SURFACE],
            note=instruction + "（没有任何证据链 -> 正确输出是 unknown，不得编造）"))
    return CounterfactualPair(
        pair_id="cf_partial_insufficient_1", family="insufficient_evidence",
        depends_on="证据不足（ambiguous vs unknown 的区分）",
        shared_nodes=nodes, shared_edges=shared,
        models=[_model_entry(model, "cf_p4_force")], relations=RELATIONS,
        variants={"base": ambiguous, "counterfactual": unknown},
        meta={"instruction": instruction, "candidate_text_source": "surface"})


def build_lexicon() -> Dict[str, Any]:
    entries = []
    for body in ("cf_p1_body_1", "cf_p1_body_2", "cf_p2_body_1", "cf_p2_body_2",
                 "cf_p3_body_1", "cf_p3_body_2", "cf_p4_body_1", "cf_p4_body_2"):
        entries.append({"node_id": body, "zh": [SURFACE]})
    entries += [
        {"node_id": "cf_p1_slot_A", "zh": ["八针工位"]},
        {"node_id": "cf_p1_slot_B", "zh": ["十二针工位"]},
        {"node_id": "cf_p1_type_8pin", "zh": ["八针接插件"]},
        {"node_id": "cf_p1_type_12pin", "zh": ["十二针接插件"]},
        {"node_id": "cf_p2_slot_A", "zh": ["A 工位槽"]},
        {"node_id": "cf_p2_slot_B", "zh": ["B 工位槽"]},
        {"node_id": "cf_p3_slot_A", "zh": ["A 工位槽"]},
        {"node_id": "cf_p4_slot_A", "zh": ["A 工位槽"]},
        {"node_id": "cf_p1_step1", "zh": ["第一步"]},
        {"node_id": "cf_p2_step1", "zh": ["第一步"]},
        {"node_id": "cf_p2_step2", "zh": ["第二步"]},
        {"node_id": "cf_p3_step1", "zh": ["第一步"]},
        {"node_id": "cf_p4_step1", "zh": ["第一步"]},
        {"node_id": "insert", "zh": ["插"]},
        {"node_id": "cf_p1_seated", "zh": ["到位"]},
        {"node_id": "cf_p2_seated", "zh": ["到位"]},
        {"node_id": "cf_p3_seated", "zh": ["到位"]},
        {"node_id": "cf_p4_seated", "zh": ["到位"]},
        {"node_id": "cf_p1_force", "zh": ["压紧力按 15–20N"]},
        {"node_id": "cf_p2_force", "zh": ["压紧力按 15–20N"]},
        {"node_id": "cf_p3_force", "zh": ["压紧力按 15–20N"]},
        {"node_id": "cf_p4_force", "zh": ["压紧力按 15–20N"]},
    ]
    payload = {"meta": {"synthetic": True,
                        "purpose": "部分可观测反事实数据集专用词表",
                        "note": "合成数据"},
               "entries": entries}
    write_json(OUT_LEXICON, payload)
    return payload


def main() -> int:
    dataset = CounterfactualDataset(
        name="counterfactual_partial",
        pairs=[build_p1(), build_p2(), build_p3(), build_p4()],
        meta={
            "synthetic": True,
            "purpose": ("部分可观测反事实：分别制造"
                        "①间接可推断 ②多跳可推断 ③来源冲突 ④证据不足，"
                        "要求正确区分『唯一可推断 / 多候选 / 缺证据 / 冲突』"),
            "leakage_control": "同既有反事实数据集：同 pair 各变体共享节点表与指令",
            "note": "合成数据；不得作为真实产品资料引用",
        },
    )
    dataset.save(OUT)
    lex = build_lexicon()
    report = validate_counterfactual_dataset(dataset)
    probe = leakage_probe(dataset)

    print("=" * 78)
    print("部分可观测反事实数据集（P1–P4）")
    print("=" * 78)
    for pair in dataset.pairs:
        statuses = {vid: v.gold.expected_status for vid, v in pair.variants.items()}
        print(f"  {pair.pair_id:<26} 族={pair.family:<22} 变体状态={statuses}")
    print()
    print(report.format_text())
    print()
    print("泄漏捷径量化（成对正确率；越接近 0 越好）：")
    for key, value in probe["shortcut_pairwise_accuracy"].items():
        print(f"    {key:<16} {value}")
    print(f"    探针 pair 数 = {probe['n_pairs_probed']}"
          f"（P3/P4 的 gold 非唯一，不参与成对翻转统计）")
    print(f"\n  词表已写入: {OUT_LEXICON}（{len(lex['entries'])} 条）")
    print(f"  已写入: {OUT}")

    write_json(_WORKSPACE / "outputs" / "counterfactual_partial_validation.json",
               {"validation": report.to_dict(), "leakage_probe": probe})
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
