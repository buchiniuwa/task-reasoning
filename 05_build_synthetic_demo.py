#!/usr/bin/env python
"""构建**合成**演示数据（明确标注 synthetic，不是真实装配资料）。

本脚本里的节点表、关系表、样本表全部是**手工写死的清单**；
脚本只做三件事：序列化、补齐交叉引用（三元组 id）、自校验。
它不生成随机数据、不做批量扩充、不伪造真实 BOM 或工艺参数。

产出：
    data/synthetic/kg_electronic_demo.json    多型号装配知识图谱（4 个型号）
    data/synthetic/intent_demo.json           意图样本 + 困难负样本对

设计意图（对应 docs/research1_dataset_design.md 的第 4/5/6 节）：
  * 覆盖两个装配类别（异形接插件插装、CPU 散热模组安装），
    第三个类别（异形功能模块多工序装配）按同样模式在正式生成阶段补齐；
  * 图谱里刻意留出两种"真实的缺陷"：
      - COOL_Y2_A 缺少 constrained_by 边  -> 证据缺失样本（MISSING）
      - CONN_X1_A 有两个满支持见证操作     -> 指代不明样本（AMBIGUOUS）
  * 困难负样本覆盖 N1–N6，并显式标注"硬掩码之后是否仍困难"。

用法：
    python scripts/05_build_synthetic_demo.py            # 生成 + 自校验
    python scripts/05_build_synthetic_demo.py --check     # 只校验已有文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.splits import (                      # noqa: E402
    PROTOCOL_SPECS, SplitProtocol, assign_split, check_leakage, find_near_duplicates,
)
from assemmate_ext.kg.consistency import check_consistency       # noqa: E402
from assemmate_ext.kg.intent import IntentDatasetSlice           # noqa: E402
from assemmate_ext.kg.intent_validate import validate_intent_dataset  # noqa: E402
from assemmate_ext.kg.io import read_document, write_document    # noqa: E402
from assemmate_ext.kg.validate import validate_document          # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout        # noqa: E402
from assemmate_ext.utils.jsonio import read_json, write_json     # noqa: E402

setup_utf8_stdout()

KG_PATH = _WORKSPACE / "data" / "synthetic" / "kg_electronic_demo.json"
INTENT_PATH = _WORKSPACE / "data" / "synthetic" / "intent_demo.json"

SYNTH_NOTE = (
    "本文件为**合成**演示数据，用于验证数据格式与校验接口；"
    "不是真实产品的 BOM、工艺参数或装配说明书，不得用于任何性能结论。"
)


# ====================================================================== 图谱
class GraphBuilder:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.nodes: List[Dict[str, Any]] = []
        self.triples: List[Dict[str, Any]] = []
        self.models: List[Dict[str, Any]] = []
        self._seq = 0
        self._edge_index: Dict[Tuple[str, str], str] = {}

    def node(self, nid: str, name: str, ntype: str, model: Optional[str],
             ref: str, extra: Optional[Dict[str, Any]] = None) -> str:
        self.nodes.append({
            "id": nid, "name": name, "node_type": ntype, "model_id": model,
            "aliases": [], "state": {"assembly_state": "unknown"},
            "evidence": {"source": "manual_annotation", "ref": ref,
                         "note": "合成演示数据（非真实产品资料）"},
            "extra": {"synthetic": True, **(extra or {})},
        })
        return nid

    def link(self, head: str, relation: str, tail: str, model: Optional[str],
             ref: str, note: str = "") -> str:
        self._seq += 1
        tid = f"t{self._seq:03d}"
        self.triples.append({
            "id": tid, "head": head, "relation": relation, "tail": tail,
            "model_id": model,
            "evidence": {"source": "manual_annotation", "ref": ref,
                         "note": note or "合成演示数据（非真实产品资料）"},
            "state": {"assembly_state": "unknown"},
        })
        self._edge_index[(head, relation)] = tid
        return tid

    def edge_id(self, head: str, relation: str) -> Optional[str]:
        return self._edge_index.get((head, relation))

    def model(self, model_id: str, family: str, display: str,
              component_types: List[str], constraint_ids: List[str], ref: str) -> None:
        self.models.append({
            "model_id": model_id, "display_name": display, "family": family,
            "description": f"合成演示型号（{family}）",
            "component_types": component_types, "constraint_ids": constraint_ids,
            "state": "not_started",
            "evidence": {"source": "manual_annotation", "ref": ref,
                         "note": "合成演示数据（非真实产品资料）"},
            "extra": {"synthetic": True},
        })

    def to_dict(self) -> Dict[str, Any]:
        from assemmate_ext.kg.schema import LEGACY_RELATION_MAP  # noqa: F401  (保持引用一致)

        relations = [
            ("instance_of", "部件实例 -> 部件类型", False),
            ("has_component", "产品型号 -> 部件实例", False),
            ("has_step", "产品型号 -> 工序", False),
            ("has_operation", "工序 -> 操作", False),
            ("has_action", "操作 -> 动作", False),
            ("acts_on", "操作 -> 作用对象", False),
            ("acts_at", "操作 -> 作用位置", False),
            ("achieves", "操作 -> 目标状态", False),
            ("constrained_by", "操作 -> 工艺约束", False),
            ("requires_tool", "操作 -> 工具", False),
            ("has_detail", "操作 -> 细节/状态/约束说明", False),
            ("has_type", "节点 -> 类型/属性", False),
            ("compatible_with", "跨型号兼容（显式允许跨型号）", True),
        ]
        used = {t["relation"] for t in self.triples}
        return {
            "schema_version": "0.1.0",
            "doc_id": self.doc_id,
            "state": "not_started",
            "meta": {
                "synthetic": True,
                "purpose": SYNTH_NOTE,
                "families": sorted({m["family"] for m in self.models}),
                "n_models": len(self.models),
                "notes": [
                    "COOL_Y2_A 刻意缺少 constrained_by 边，用于制造合法的『证据缺失』样本。",
                    "CONN_X1_A 的 op1_1 与 op1_3 除了作用对象外边完全相同，"
                    "构成两个满支持见证操作，用于制造合法的『指代不明』样本。",
                    "工序顺序用 step 编号表达；next 边留给正式数据集。",
                ],
            },
            "models": self.models,
            "nodes": self.nodes,
            "relations": [
                {"name": name, "relation_type": name, "domain": [], "range": [],
                 "description": desc, "cross_model": cross,
                 "evidence": {"source": "manual_annotation", "ref": self.doc_id}}
                for name, desc, cross in relations if name in used or name == "compatible_with"
            ],
            "triples": self.triples,
        }


def build_kg() -> Dict[str, Any]:
    g = GraphBuilder("kg_electronic_demo")

    # ---------------------------------------------------------- 全局共享节点
    t = "SYN-GLOBAL"
    g.node("connector_body", "connector_body", "component_type", None, t)
    g.node("guide_pin_type", "guide_pin_type", "component_type", None, t)
    g.node("pin_header_type", "pin_header_type", "component_type", None, t)
    g.node("heatsink_type", "heatsink_type", "component_type", None, t)
    g.node("insert", "insert", "action", None, t)
    g.node("press", "press", "action", None, t)
    g.node("place", "place", "action", None, t)
    g.node("tighten", "tighten", "action", None, t)
    g.node("insertion_tool", "insertion_tool", "tool", None, t)
    g.node("torque_wrench", "torque_wrench", "tool", None, t)
    g.node("graspable", "graspable", "attribute", None, t)

    # ---------------------------------------------------------- 类别一：异形接插件
    def connector_variant(p: str, model: str, ref: str,
                          force: str, force_val: str) -> None:
        g.node(f"{p}_model", model, "model", model, ref)
        g.node(f"{p}_step1", f"{model}_STEP1", "step", model, ref)
        g.node(f"{p}_step2", f"{model}_STEP2", "step", model, ref)
        g.node(f"{p}_body", f"ConnectorBody_{model}", "component_instance", model, ref)
        g.node(f"{p}_guide_pin", f"GuidePin_{model}", "component_instance", model, ref)
        g.node(f"{p}_latch", f"Latch_{model}", "component_instance", model, ref)
        g.node(f"{p}_slot", f"{model}_SLOT_A", "location", model, ref)
        g.node(f"{p}_slot_b", f"{model}_SLOT_B", "location", model, ref)
        g.node(f"{p}_seated", f"{model}_SEATED", "target_state", model, ref)
        g.node(f"{p}_flush", f"{model}_FLUSH", "target_state", model, ref)
        g.node(f"{p}_force", force_val, "process_constraint", model, ref)

        g.link(f"{p}_model", "has_step", f"{p}_step1", model, ref)
        g.link(f"{p}_model", "has_component", f"{p}_body", model, ref)
        g.link(f"{p}_model", "has_component", f"{p}_guide_pin", model, ref)
        g.link(f"{p}_body", "instance_of", "connector_body", model, ref)
        g.link(f"{p}_guide_pin", "instance_of", "guide_pin_type", model, ref)
        g.link(f"{p}_body", "has_type", "graspable", model, ref)

    # -- CONN_X1_A：含 op1_1 / op1_3 两个"满支持"见证操作 + step2
    p, m, ref = "conn_x1_a", "CONN_X1_A", "SYN-ANN-CX1A-01"
    connector_variant(p, m, ref, "force_15_20n", f"{m}_FORCE_15_20N")
    g.node(f"{p}_torque", f"{m}_TORQUE_1_2NM", "process_constraint", m, ref)
    g.link(f"{p}_model", "has_step", f"{p}_step2", m, ref)
    # 接口模型（N7 错误接口负样本的依据）：主体/定位销与该工位**有**兼容证据，
    # 排针与该工位**没有** compatible_with 证据 -> 构成"疑似接口不匹配"。
    # 注意这是开放世界判定：无证据 ≠ 已证明不兼容（见 dataset_design §5.3）。
    g.node(f"{p}_pin_header", f"PinHeader_{m}", "component_instance", m, ref)
    g.link(f"{p}_pin_header", "instance_of", "pin_header_type", m, ref)
    g.link(f"{p}_body", "compatible_with", f"{p}_slot", m, ref,
           note="接口兼容证据：主体可插入该工位")
    g.link(f"{p}_guide_pin", "compatible_with", f"{p}_slot", m, ref,
           note="接口兼容证据：定位销可插入该工位")

    def op(prefix: str, opname: str, model: str, action: str, obj: str,
           loc: str, state: str, constraint: str, tool: str, ref: str) -> str:
        op_id = f"{prefix}_{opname}"
        g.node(op_id, f"{model}_{opname.upper()}", "operation", model, ref)
        g.link(op_id, "has_action", action, model, ref)
        g.link(op_id, "acts_on", obj, model, ref)
        g.link(op_id, "acts_at", loc, model, ref)
        g.link(op_id, "achieves", state, model, ref)
        g.link(op_id, "constrained_by", constraint, model, ref)
        g.link(op_id, "requires_tool", tool, model, ref)
        return op_id

    op1_1 = op(p, "op1_1", m, "insert", f"{p}_body", f"{p}_slot",
               f"{p}_seated", f"{p}_force", "insertion_tool", ref)
    op1_3 = op(p, "op1_3", m, "insert", f"{p}_guide_pin", f"{p}_slot",
               f"{p}_seated", f"{p}_force", "insertion_tool", ref)
    op2_1 = op(p, "op2_1", m, "press", f"{p}_latch", f"{p}_slot_b",
               f"{p}_flush", f"{p}_torque", "insertion_tool", ref)
    g.link(f"{p}_step1", "has_operation", op1_1, m, ref)
    g.link(f"{p}_step1", "has_operation", op1_3, m, ref)
    g.link(f"{p}_step2", "has_operation", op2_1, m, ref)

    # -- CONN_X1_B：与 A 共享部件类型（跨型号同名负样本的来源）
    # 注意：B 的 step2/锁扣/B 工位槽都必须真正参与三元组，否则会产生孤立节点（W001）
    pb, mb, refb = "conn_x1_b", "CONN_X1_B", "SYN-ANN-CX1B-01"
    connector_variant(pb, mb, refb, "force_20_25n", f"{mb}_FORCE_20_25N")
    g.link(f"{pb}_model", "has_step", f"{pb}_step2", mb, refb)
    opb = op(pb, "op1_1", mb, "insert", f"{pb}_body", f"{pb}_slot",
             f"{pb}_seated", f"{pb}_force", "insertion_tool", refb)
    opb2 = op(pb, "op2_1", mb, "press", f"{pb}_latch", f"{pb}_slot_b",
              f"{pb}_flush", f"{pb}_force", "insertion_tool", refb)
    g.link(f"{pb}_step1", "has_operation", opb, mb, refb)
    g.link(f"{pb}_step2", "has_operation", opb2, mb, refb)

    # ---------------------------------------------------------- 类别二：CPU 散热模组
    pc, mc, refc = "cool_y1_a", "COOL_Y1_A", "SYN-ANN-CY1A-01"
    g.node(f"{pc}_model", mc, "model", mc, refc)
    g.node(f"{pc}_step1", f"{mc}_STEP1", "step", mc, refc)
    g.node(f"{pc}_heatsink", f"Heatsink_{mc}", "component_instance", mc, refc)
    g.node(f"{pc}_cpu", f"CPU_{mc}", "component_instance", mc, refc)
    g.node(f"{pc}_socket", f"{mc}_SOCKET", "location", mc, refc)
    g.node(f"{pc}_frame", f"{mc}_FRAME", "location", mc, refc)
    g.node(f"{pc}_seated", f"{mc}_SEATED", "target_state", mc, refc)
    g.node(f"{pc}_paste_even", f"{mc}_PASTE_EVEN", "target_state", mc, refc)
    g.node(f"{pc}_torque", f"{mc}_TORQUE_8_10NM", "process_constraint", mc, refc)
    g.node(f"{pc}_paste_dot", f"{mc}_PASTE_DOT_SPEC", "process_constraint", mc, refc)
    g.link(f"{pc}_model", "has_step", f"{pc}_step1", mc, refc)
    g.link(f"{pc}_model", "has_component", f"{pc}_heatsink", mc, refc)
    g.link(f"{pc}_model", "has_component", f"{pc}_cpu", mc, refc)
    g.link(f"{pc}_heatsink", "instance_of", "heatsink_type", mc, refc)
    g.link(f"{pc}_heatsink", "has_type", "graspable", mc, refc)
    opc1 = op(pc, "op1_1", mc, "place", f"{pc}_heatsink", f"{pc}_socket",
              f"{pc}_seated", f"{pc}_torque", "torque_wrench", refc)
    opc2 = op(pc, "op1_2", mc, "tighten", f"{pc}_cpu", f"{pc}_frame",
              f"{pc}_paste_even", f"{pc}_paste_dot", "torque_wrench", refc)
    g.link(f"{pc}_step1", "has_operation", opc1, mc, refc)
    g.link(f"{pc}_step1", "has_operation", opc2, mc, refc)

    # -- COOL_Y2_A：**刻意缺少 constrained_by**（证据缺失样本的来源）
    pd, md, refd = "cool_y2_a", "COOL_Y2_A", "SYN-ANN-CY2A-01"
    g.node(f"{pd}_model", md, "model", md, refd)
    g.node(f"{pd}_step1", f"{md}_STEP1", "step", md, refd)
    g.node(f"{pd}_heatsink", f"Heatsink_{md}", "component_instance", md, refd)
    g.node(f"{pd}_socket", f"{md}_SOCKET", "location", md, refd)
    g.node(f"{pd}_seated", f"{md}_SEATED", "target_state", md, refd)
    g.node(f"{pd}_detail", f"{md}_TORQUE_DETAIL_NOTE", "process_constraint", md, refd)
    g.link(f"{pd}_model", "has_step", f"{pd}_step1", md, refd)
    g.link(f"{pd}_model", "has_component", f"{pd}_heatsink", md, refd)
    g.link(f"{pd}_heatsink", "instance_of", "heatsink_type", md, refd)
    opd = f"{pd}_op1_1"
    g.node(opd, f"{md}_OP1_1", "operation", md, refd)
    g.link(opd, "has_action", "place", md, refd)
    g.link(opd, "acts_on", f"{pd}_heatsink", md, refd)
    g.link(opd, "acts_at", f"{pd}_socket", md, refd)
    g.link(opd, "achieves", f"{pd}_seated", md, refd)
    g.link(opd, "requires_tool", "torque_wrench", md, refd)
    # 注意：这里用 has_detail 而不是 constrained_by —— 所以"结构化的工艺约束"确实缺失
    g.link(opd, "has_detail", f"{pd}_detail", md, refd,
           note="该型号工艺参数只以细节文本形式存在，尚未结构化为 constrained_by")
    g.link(f"{pd}_step1", "has_operation", opd, md, refd)

    # ---------------------------------------------------------- 型号声明
    g.model("CONN_X1_A", "connector_insertion", "异形接插件 A 型",
            ["connector_body", "guide_pin_type", "pin_header_type"],
            [f"{p}_force", f"{p}_torque"], "SYN-ANN-CX1A-01")
    g.model("CONN_X1_B", "connector_insertion", "异形接插件 B 型",
            ["connector_body", "guide_pin_type"], [f"{pb}_force"], "SYN-ANN-CX1B-01")
    g.model("COOL_Y1_A", "cpu_cooler_mounting", "CPU 散热模组 A 型",
            ["heatsink_type"], [f"{pc}_torque", f"{pc}_paste_dot"], "SYN-ANN-CY1A-01")
    g.model("COOL_Y2_A", "cpu_cooler_mounting", "CPU 散热模组 B 型",
            ["heatsink_type"], [], "SYN-ANN-CY2A-01")

    return g.to_dict(), g


# ====================================================================== 意图样本
def build_intent(g: GraphBuilder) -> Dict[str, Any]:
    p, m = "conn_x1_a", "CONN_X1_A"
    pb, mb = "conn_x1_b", "CONN_X1_B"
    pc, mc = "cool_y1_a", "COOL_Y1_A"
    pd, md = "cool_y2_a", "COOL_Y2_A"

    def slot(slot_name: str, status: str, ids: List[str],
             surface: Optional[List[str]] = None,
             triples: Optional[List[str]] = None, note: Optional[str] = None) -> Dict[str, Any]:
        return {"slot": slot_name, "status": status, "entity_ids": list(ids),
                "surface": surface or [], "evidence_triple_ids": triples or [], "note": note}

    def edges_of(op_id: str, slots: List[str]) -> List[str]:
        """取某操作对指定槽位的证据三元组 id（交叉引用由脚本补齐，杜绝悬空引用）。"""
        rel = {"object": "acts_on", "action": "has_action", "location": "acts_at",
               "target_state": "achieves", "process_constraint": "constrained_by",
               "tool": "requires_tool"}
        out = []
        for s in slots:
            tid = g.edge_id(op_id, rel[s])
            if tid:
                out.append(tid)
        return out

    samples: List[Dict[str, Any]] = []

    # ---- S1 COMPLETE：正常样本（类别一） ----
    op1_1 = f"{p}_op1_1"
    samples.append({
        "sample_id": "s_conn_x1a_complete_01",
        "product_family": "connector_insertion",
        "model_id": m,
        "instruction": "把接插件主体插到 A 工位槽里，压到位",
        "current_state": "in_progress",
        "current_state_ops": [],
        "gold_intent": {
            "model_id": m, "goal": "完成接插件主体插装",
            "slots": {
                "object": slot("object", "grounded", [f"{p}_body"], ["接插件主体"],
                               edges_of(op1_1, ["object"])),
                "action": slot("action", "grounded", ["insert"], ["插到"],
                               edges_of(op1_1, ["action"])),
                "location": slot("location", "grounded", [f"{p}_slot"], ["A 工位槽"],
                                 edges_of(op1_1, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"], ["到位"],
                                     edges_of(op1_1, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"],
                                           ["压到位"], edges_of(op1_1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{p}_body", "insert", f"{p}_slot", f"{p}_seated", f"{p}_force"],
        "evidence_triples": edges_of(op1_1, ["object", "action", "location",
                                             "target_state", "process_constraint", "tool"]),
        "evidence_source_ids": ["SYN-ANN-CX1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "complete",
        "group_id": "g_cx1a_body_insert",
        "synthetic": True,
        "meta": {"synthetic": True},
    })

    # ---- S2 COMPLETE：正常样本（类别二，另一个改写族） ----
    opc1 = f"{pc}_op1_1"
    samples.append({
        "sample_id": "s_cool_y1a_complete_01",
        "product_family": "cpu_cooler_mounting",
        "model_id": mc,
        "instruction": "将散热片放到 CPU 底座上，确认已就位",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": mc, "goal": "安装散热片",
            "slots": {
                "object": slot("object", "grounded", [f"{pc}_heatsink"], ["散热片"],
                               edges_of(opc1, ["object"])),
                "action": slot("action", "grounded", ["place"], ["放到"],
                               edges_of(opc1, ["action"])),
                "location": slot("location", "grounded", [f"{pc}_socket"], ["CPU 底座"],
                                 edges_of(opc1, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{pc}_seated"], ["已就位"],
                                     edges_of(opc1, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{pc}_torque"],
                                           ["按扭矩规范"], edges_of(opc1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{pc}_heatsink", "place", f"{pc}_socket", f"{pc}_seated", f"{pc}_torque"],
        "evidence_triples": edges_of(opc1, ["object", "action", "location",
                                            "target_state", "process_constraint", "tool"]),
        "evidence_source_ids": ["SYN-ANN-CY1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "complete",
        "group_id": "g_cy1a_heatsink_place",
        "synthetic": True,
        "meta": {"synthetic": True},
    })

    # ---- S3 MISSING：证据缺失（COOL_Y2_A 无 constrained_by） ----
    opd = f"{pd}_op1_1"
    samples.append({
        "sample_id": "s_cool_y2a_missing_constraint",
        "product_family": "cpu_cooler_mounting",
        "model_id": md,
        "instruction": "按工艺要求把散热片装到底座上，扭矩参照规范执行",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": md, "goal": "安装散热片（扭矩规范待补录）",
            "slots": {
                "object": slot("object", "grounded", [f"{pd}_heatsink"], ["散热片"],
                               edges_of(opd, ["object"])),
                "action": slot("action", "grounded", ["place"], ["装到"],
                               edges_of(opd, ["action"])),
                "location": slot("location", "grounded", [f"{pd}_socket"], ["底座"],
                                 edges_of(opd, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{pd}_seated"], ["装好"],
                                     edges_of(opd, ["target_state"])),
                "process_constraint": slot(
                    "process_constraint", "unknown", [],
                    ["参照规范"],
                    note="该型号图谱缺少 constrained_by 边：只存在 has_detail 文本，"
                         "结构化工艺约束尚未录入 —— 必须标 unknown，不得编造数值"),
            },
        },
        "gold_entity_ids": [f"{pd}_heatsink", "place", f"{pd}_socket", f"{pd}_seated"],
        "evidence_triples": edges_of(opd, ["object", "action", "location",
                                           "target_state", "tool"]),
        "evidence_source_ids": ["SYN-ANN-CY2A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "missing_evidence",
        "conflict_slot": "process_constraint",
        "group_id": "g_cy2a_heatsink_place",
        "synthetic": True,
        "meta": {"synthetic": True, "expected_slot_status": "unknown"},
    })

    # ---- S4 CONFLICT：事实冲突（动作与对象/状态组合在图中不成立） ----
    op2_1 = f"{p}_op2_1"
    samples.append({
        "sample_id": "s_conn_x1a_conflict_action",
        "product_family": "connector_insertion",
        "model_id": m,
        "instruction": "用插入的方式把接插件锁扣压到齐平",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": m, "goal": "（含矛盾的指令）",
            "slots": {
                "object": slot("object", "grounded", [f"{p}_latch"], ["锁扣"],
                               edges_of(op2_1, ["object"])),
                "action": slot("action", "conflict", ["insert"], ["用插入的方式"],
                               note="指令要求 insert，但没有任何操作同时对 act_on=锁扣 且 achieves=齐平"),
                "location": slot("location", "grounded", [f"{p}_slot"], ["工位槽"],
                                 edges_of(op1_1, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{p}_flush"], ["齐平"],
                                     edges_of(op2_1, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"],
                                           ["压紧力"], edges_of(op1_1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{p}_latch", "insert", f"{p}_slot", f"{p}_flush", f"{p}_force"],
        "evidence_triples": sorted(set(
            edges_of(op1_1, ["location", "process_constraint"])
            + edges_of(op2_1, ["object", "target_state"])
        )),
        "evidence_source_ids": ["SYN-ANN-CX1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "conflicting_evidence",
        "conflict_slot": None,
        "group_id": "g_cx1a_latch_conflict",
        "synthetic": True,
        "meta": {
            "synthetic": True,
            "expected_diagnosis": "fact_conflict",
            # 这是一个**多冲突**样本：3 个槽位来自 op1_1、2 个来自 op2_1，
            # 两两之间都不相容（6 个最小冲突集，交集为空）。
            # 因此**不允许**声明唯一的 conflict_slot —— 算法给出的是
            # 两个最小修复集（改哪一组都行），而不是"某一个槽位错了"。
            "multi_conflict": True,
            "expected_repair_sets": 2,
        },
    })

    # ---- S4b CONFLICT（唯一可归责）：只有 action 是异常项 ----
    # 与 S4 对照：S4 是"两组互相矛盾"（不可唯一归责），这里是"单槽位异常"
    # （最小冲突集交集 = {action}，可唯一归责）。两种情形必须分开统计。
    samples.append({
        "sample_id": "s_conn_x1a_conflict_action_unique",
        "product_family": "connector_insertion",
        "model_id": m,
        "instruction": "用压紧的方式把接插件主体插到工位槽里并到位",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": m, "goal": "（动作与其余槽位不相容）",
            "slots": {
                "object": slot("object", "grounded", [f"{p}_body"], ["接插件主体"],
                               edges_of(op1_1, ["object"])),
                "action": slot("action", "conflict", ["press"], ["用压紧的方式"],
                               note="指令给出的动作是 press，但没有任何操作对"
                                    "『主体 + 该工位 + 就位 + 该压紧力』同时成立"),
                "location": slot("location", "grounded", [f"{p}_slot"], ["工位槽"],
                                 edges_of(op1_1, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"], ["到位"],
                                     edges_of(op1_1, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"],
                                           ["压紧力"], edges_of(op1_1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{p}_body", "press", f"{p}_slot", f"{p}_seated", f"{p}_force"],
        "evidence_triples": edges_of(op1_1, ["object", "location", "target_state",
                                            "process_constraint"]),
        "evidence_source_ids": ["SYN-ANN-CX1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "conflicting_evidence",
        "conflict_slot": "action",
        "group_id": "g_cx1a_action_conflict_unique",
        "synthetic": True,
        "meta": {"synthetic": True, "expected_diagnosis": "fact_conflict",
                 "expected_localized_slots": ["action"]},
    })

    # ---- S5 STATE_CONTRADICTION：当前状态已经是目标状态 ----
    samples.append({
        "sample_id": "s_cool_y1a_state_regression",
        "product_family": "cpu_cooler_mounting",
        "model_id": mc,
        "instruction": "把散热片装到底座上让它就位",
        "current_state": "in_progress",
        "current_state_node": f"{pc}_seated",
        "gold_intent": {
            "model_id": mc, "goal": "（目标状态其实已满足）",
            "slots": {
                "object": slot("object", "grounded", [f"{pc}_heatsink"], ["散热片"],
                               edges_of(opc1, ["object"])),
                "action": slot("action", "grounded", ["place"], ["装到"],
                               edges_of(opc1, ["action"])),
                "location": slot("location", "grounded", [f"{pc}_socket"], ["底座"],
                                 edges_of(opc1, ["location"])),
                "target_state": slot("target_state", "conflict", [f"{pc}_seated"], ["就位"],
                                     note="当前状态节点与目标状态是同一个节点："
                                          "要求达到的状态已经满足，构成状态矛盾"),
                "process_constraint": slot("process_constraint", "grounded", [f"{pc}_torque"],
                                           ["按扭矩规范"], edges_of(opc1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{pc}_heatsink", "place", f"{pc}_socket", f"{pc}_seated", f"{pc}_torque"],
        "evidence_triples": edges_of(opc1, ["object", "action", "location",
                                            "target_state", "process_constraint", "tool"]),
        "evidence_source_ids": ["SYN-ANN-CY1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "state_contradiction",
        "conflict_slot": "target_state",
        "group_id": "g_cy1a_state_regression",
        "synthetic": True,
        "meta": {"synthetic": True, "expected_diagnosis": "fact_conflict"},
    })

    # ---- S6 AMBIGUOUS_REFERENCE：两个满支持见证操作 ----
    op1_3 = f"{p}_op1_3"
    samples.append({
        "sample_id": "s_conn_x1a_ambiguous_reference",
        "product_family": "connector_insertion",
        "model_id": m,
        "instruction": "把那个零件插到工位槽里到位",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": m, "goal": "（指代不明：无法判定是哪个零件）",
            "slots": {
                "object": slot("object", "ambiguous", [f"{p}_body", f"{p}_guide_pin"],
                               ["那个零件"],
                               note="两个零件都能被某种操作完整见证，无法区分"),
                "action": slot("action", "grounded", ["insert"], ["插到"],
                               edges_of(op1_1, ["action"])),
                "location": slot("location", "grounded", [f"{p}_slot"], ["工位槽"],
                                 edges_of(op1_1, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"], ["到位"],
                                     edges_of(op1_1, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"],
                                           ["压到位"], edges_of(op1_1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{p}_body", f"{p}_guide_pin", "insert", f"{p}_slot",
                            f"{p}_seated", f"{p}_force"],
        "evidence_triples": sorted(set(
            edges_of(op1_1, ["object", "action", "location", "target_state", "process_constraint"])
            + edges_of(op1_3, ["object"])
        )),
        "evidence_source_ids": ["SYN-ANN-CX1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "ambiguous_reference",
        "group_id": "g_cx1a_ambiguous_part",
        "alternatives": [
            {"model_id": m, "slots": {
                "object": slot("object", "grounded", [f"{p}_body"]),
                "action": slot("action", "grounded", ["insert"]),
                "location": slot("location", "grounded", [f"{p}_slot"]),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"]),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"]),
            }},
            {"model_id": m, "slots": {
                "object": slot("object", "grounded", [f"{p}_guide_pin"]),
                "action": slot("action", "grounded", ["insert"]),
                "location": slot("location", "grounded", [f"{p}_slot"]),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"]),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"]),
            }},
        ],
        "synthetic": True,
        "meta": {"synthetic": True, "expected_diagnosis": "ambiguous"},
    })

    # ---- S7 MULTI_MODEL_AMBIGUITY：型号约束消歧 ----
    samples.append({
        "sample_id": "s_conn_x1a_multimodel",
        "product_family": "connector_insertion",
        "model_id": m,
        "instruction": "接插件主体要装到位",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": m, "goal": "在 CONN_X1_A 型号上下文中装接插件主体",
            "slots": {
                "object": slot("object", "grounded", [f"{p}_body"], ["接插件主体"],
                               edges_of(op1_1, ["object"])),
                "action": slot("action", "grounded", ["insert"], ["装"],
                               edges_of(op1_1, ["action"])),
                "location": slot("location", "grounded", [f"{p}_slot"], ["工位槽"],
                                 edges_of(op1_1, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"], ["到位"],
                                     edges_of(op1_1, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"],
                                           ["压到位"], edges_of(op1_1, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{p}_body", "insert", f"{p}_slot", f"{p}_seated", f"{p}_force"],
        "evidence_triples": edges_of(op1_1, ["object", "action", "location",
                                             "target_state", "process_constraint", "tool"]),
        "evidence_source_ids": ["SYN-ANN-CX1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "multi_model_ambiguity",
        "group_id": "g_cx1a_multimodel_part",
        "synthetic": True,
        "meta": {
            "synthetic": True,
            "model_sensitive": True,
            "distractor_entity_ids": [f"{pb}_body"],
            "note": "『接插件主体』在 A/B 两个型号都存在；只有型号条件能消歧",
        },
    })

    # ---- S8 COMPLETE + 同型号相似实体（供 P3 划分使用） ----
    samples.append({
        "sample_id": "s_conn_x1a_similar_entity",
        "product_family": "connector_insertion",
        "model_id": m,
        "instruction": "把定位销插到工位槽里到位",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": m, "goal": "安装定位销",
            "slots": {
                "object": slot("object", "grounded", [f"{p}_guide_pin"], ["定位销"],
                               edges_of(op1_3, ["object"])),
                "action": slot("action", "grounded", ["insert"], ["插到"],
                               edges_of(op1_3, ["action"])),
                "location": slot("location", "grounded", [f"{p}_slot"], ["工位槽"],
                                 edges_of(op1_3, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{p}_seated"], ["到位"],
                                     edges_of(op1_3, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{p}_force"],
                                           ["压到位"], edges_of(op1_3, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{p}_guide_pin", "insert", f"{p}_slot", f"{p}_seated", f"{p}_force"],
        "evidence_triples": edges_of(op1_3, ["object", "action", "location",
                                             "target_state", "process_constraint", "tool"]),
        "evidence_source_ids": ["SYN-ANN-CX1A-01"],
        "annotation_source": "manual_annotation",
        "scenario": "complete",
        "group_id": "g_cx1a_guide_pin_insert",
        "synthetic": True,
        "meta": {
            "synthetic": True,
            "similar_entity_case": True,
            "distractor_entity_ids": [f"{p}_body"],   # 同型号内的相似零件
        },
    })

    # ---- S9 COMPLETE（型号 CONN_X1_B）：作为 P2「未见型号」的被留出型号 ----
    opb = f"{pb}_op1_1"
    samples.append({
        "sample_id": "s_conn_x1b_complete_01",
        "product_family": "connector_insertion",
        "model_id": mb,
        "instruction": "接插件主体插入 A 工位槽并确认就位",
        "current_state": "in_progress",
        "gold_intent": {
            "model_id": mb, "goal": "在 CONN_X1_B 上完成接插件主体插装",
            "slots": {
                "object": slot("object", "grounded", [f"{pb}_body"], ["接插件主体"],
                               edges_of(opb, ["object"])),
                "action": slot("action", "grounded", ["insert"], ["插入"],
                               edges_of(opb, ["action"])),
                "location": slot("location", "grounded", [f"{pb}_slot"], ["A 工位槽"],
                                 edges_of(opb, ["location"])),
                "target_state": slot("target_state", "grounded", [f"{pb}_seated"], ["就位"],
                                     edges_of(opb, ["target_state"])),
                "process_constraint": slot("process_constraint", "grounded", [f"{pb}_force"],
                                           ["按压力规范"], edges_of(opb, ["process_constraint"])),
            },
        },
        "gold_entity_ids": [f"{pb}_body", "insert", f"{pb}_slot", f"{pb}_seated", f"{pb}_force"],
        "evidence_triples": edges_of(opb, ["object", "action", "location",
                                           "target_state", "process_constraint"]),
        "evidence_source_ids": ["SYN-ANN-CX1B-01"],
        "annotation_source": "manual_annotation",
        "scenario": "complete",
        "group_id": "g_cx1b_body_insert",
        "synthetic": True,
        "meta": {
            "synthetic": True,
            "similar_entity_case": True,   # 同型号内 body / guide_pin 易混，供 P3 使用
            "distractor_entity_ids": [f"{pb}_guide_pin"],
        },
    })

    # ---------------------------------------------------------------- 负样本对
    def neg(pair_id: str, sample_id: str, mention: str, slot_name: str, model: str,
            pos: str, negat: str, neg_type: str, difficulty: float,
            note: str = "") -> Dict[str, Any]:
        return {
            "pair_id": pair_id, "sample_id": sample_id, "mention": mention,
            "slot": slot_name, "model_id": model,
            "positive_id": pos, "negative_id": negat,
            "negative_type": neg_type, "difficulty": difficulty,
            "synthetic": True,
            "meta": {"synthetic": True, "note": note} if note else {"synthetic": True},
        }

    negatives = [
        neg("n_cx1a_cross_model_body", "s_conn_x1a_multimodel", "接插件主体", "object", m,
            f"{p}_body", f"{pb}_body", "n1_cross_model_homonym", 0.0,
            "会被硬性型号掩码直接过滤：单独报告，不计入同型号难度"),
        neg("n_cx1a_similar_part", "s_conn_x1a_similar_entity", "定位销", "object", m,
            f"{p}_guide_pin", f"{p}_body", "n2_same_model_similar", 1.0,
            "同型号相似零件：掩码后仍然困难，是主要战场"),
        neg("n_cx1a_wrong_location", "s_conn_x1a_complete_01", "工位槽", "location", m,
            f"{p}_slot", f"{p}_slot_b", "n3_wrong_location", 1.0,
            "型号正确、类型正确，但位置错误"),
        neg("n_cx1a_wrong_relation", "s_conn_x1a_complete_01", "把接插件主体插到位", "object", m,
            f"{p}_body", f"{p}_force", "n4_wrong_relation", 0.5,
            "关系角色错配：工艺约束节点被当成作用对象（也是类型约束的边界样本）"),
        neg("n_cx1a_object_tool", "s_conn_x1a_complete_01", "把接插件主体插到位", "object", m,
            f"{p}_body", "insertion_tool", "n5_object_tool_confusion", 0.4,
            "工具/对象混淆；仅靠类型约束即可排除，属较易负样本"),
        neg("n_cx1a_state_mismatch", "s_conn_x1a_complete_01", "到位", "target_state", m,
            f"{p}_seated", f"{p}_flush", "n6_state_mismatch", 1.0,
            "同型号内两个目标状态混淆"),
        neg("n_cy1a_similar_part", "s_cool_y1a_complete_01", "散热片", "object", mc,
            f"{pc}_heatsink", f"{pc}_cpu", "n2_same_model_similar", 1.0,
            "散热片 vs CPU：同型号内相邻部件"),
        neg("n_cy1a_state_mismatch", "s_cool_y1a_complete_01", "已就位", "target_state", mc,
            f"{pc}_seated", f"{pc}_paste_even", "n6_state_mismatch", 1.0,
            "同型号内『就位』与『硅脂均匀』两个状态混淆"),
        # 留出型号（CONN_X1_B）内部的困难负样本：P2 下模型只能靠图结构理解这个新型号
        neg("n_cx1b_similar_part", "s_conn_x1b_complete_01", "接插件主体", "object", mb,
            f"{pb}_body", f"{pb}_guide_pin", "n2_same_model_similar", 1.0,
            "未见型号内部的相似零件：型号条件必须由图结构归纳而来"),
        neg("n_cx1b_wrong_location", "s_conn_x1b_complete_01", "A 工位槽", "location", mb,
            f"{pb}_slot", f"{pb}_slot_b", "n3_wrong_location", 1.0,
            "未见型号内部的位置混淆"),
        # N7 错误接口匹配：正样本与工位之间有 compatible_with 证据，负样本没有。
        # 注意这是**开放世界**判定：缺少证据只表示"无证据"，不等于已证明不兼容，
        # 因此 N7 必须与其它类型分开报告（校验规则 I027 会检查证据依据是否存在）。
        neg("n_cx1a_wrong_interface", "s_conn_x1a_complete_01", "接插件主体", "object", m,
            f"{p}_body", f"{p}_pin_header", "n7_wrong_interface", 1.0,
            "同型号、同类型合法的部件，但与目标工位没有接口兼容证据"),
    ]

    # ---------------------------------------------------------------- 工具字段
    # 工具是工艺约束组内的**结构化字段**（Q6），由证据链中的 requires_tool 反查得到，
    # 不手工填写以免与图谱不一致；证据链里没有该边的样本保持空（不猜）。
    triple_by_id = {t["id"]: t for t in g.triples}

    def tools_of_evidence(tids):
        return sorted({
            triple_by_id[tid]["tail"]
            for tid in tids
            if tid in triple_by_id and triple_by_id[tid]["relation"] == "requires_tool"
        })

    for s in samples:
        s["gold_intent"]["tool_ids"] = tools_of_evidence(s.get("evidence_triples") or [])

    return {
        "name": "synthetic_intent_demo",
        "meta": {
            "synthetic": True,
            "purpose": SYNTH_NOTE,
            "kg_source": "data/synthetic/kg_electronic_demo.json",
            "n_samples": len(samples),
            "n_negatives": len(negatives),
            "covered_scenarios": sorted({s["scenario"] for s in samples}),
            "covered_negative_types": sorted({n["negative_type"] for n in negatives}),
            "field_spec": "见 docs/research1_dataset_design.md 第 5 节（任务 A2）",
        },
        "samples": samples,
        "negatives": negatives,
    }


# ====================================================================== 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="构建合成演示数据并自校验")
    parser.add_argument("--check", action="store_true", help="只校验已有文件，不重新生成")
    args = parser.parse_args()

    if args.check:
        if not KG_PATH.is_file() or not INTENT_PATH.is_file():
            print(f"[FAIL] 演示数据不存在，请先运行本脚本（不带 --check）", file=sys.stderr)
            return 1
        doc = read_document(KG_PATH)
    else:
        kg_raw, _builder = build_kg()
        write_json(KG_PATH, kg_raw)
        doc = read_document(KG_PATH)
        # 意图样本需要图谱的边索引，重新构建一次以获得 builder
        _, g = build_kg()
        write_json(INTENT_PATH, build_intent(g))

    print("=" * 74)
    print("第 1 部分：知识图谱校验")
    print("=" * 74)
    kg_report = validate_document(doc)
    print(f"  doc_id={doc.doc_id}  型号={doc.available_models()}")
    print(f"  节点={len(doc.nodes)}  关系={len(doc.relations)}  三元组={len(doc.triples)}")
    print(kg_report.format_text())

    print()
    print("=" * 74)
    print("第 2 部分：意图样本 + 负样本校验")
    print("=" * 74)
    data = IntentDatasetSlice.from_dict(read_json(INTENT_PATH))
    intent_report = validate_intent_dataset([doc], data)
    print(intent_report.format_text())

    print()
    print("=" * 74)
    print("第 3 部分：金标准一致性（JWC 参考实现跑在 gold 上）")
    print("=" * 74)
    for sample in data.samples:
        rep = check_consistency(
            doc, sample.gold_intent,
            current_state=sample.current_state,
            current_state_node=sample.current_state_node,
            model_id=sample.model_id,
        )
        mark = "OK " if rep.diagnosis in _expected(sample) else "!! "
        print(f"  {mark}{sample.sample_id:<34} 场景={sample.scenario:<22} "
              f"诊断={rep.diagnosis:<20} 满支持见证={rep.n_full_witnesses} "
              f"极大子集={rep.maximal_supported_subsets}")

    print()
    print("=" * 74)
    print("第 4 部分：四套划分协议（演示规模，规模不足会给出警告）")
    print("=" * 74)
    for protocol, spec in PROTOCOL_SPECS.items():
        if protocol is SplitProtocol.P2_UNSEEN_MODEL_KG_VISIBLE:
            spec = SplitSpecWithHoldout(spec, ["CONN_X1_B"])
        result = assign_split(data.samples, spec)
        leak = check_leakage(result, data)
        print(f"  {protocol.value:<32} train={len(result.train):>2} "
              f"dev={len(result.dev):>2} test={len(result.test):>2}  "
              f"泄漏={'无' if leak.ok else '有'}")
        for w in result.warnings:
            print(f"      - {w}")
        for issue in leak.issues:
            print(f"      ! {issue}")

    near = find_near_duplicates(data.samples, threshold=0.8)
    if near:
        print(f"  词面近似对（供人工确认是否漏标 group_id）: {near}")

    ok = kg_report.ok and intent_report.ok
    print()
    print(f"结论: 图谱 {'通过' if kg_report.ok else '未通过'} / "
          f"意图数据 {'通过' if intent_report.ok else '未通过'}")
    return 0 if ok else 1


def _expected(sample) -> set:
    """该样本的期望诊断（用于打印自查标记，不作评测）。"""
    scenario = sample.scenario
    if scenario in ("conflicting_evidence", "state_contradiction"):
        return {"fact_conflict"}
    if scenario == "ambiguous_reference":
        return {"ambiguous"}
    return {"consistent", "ambiguous"}


def SplitSpecWithHoldout(spec, models):
    from assemmate_ext.dataset.splits import SplitSpec

    return SplitSpec(
        protocol=spec.protocol, seed=spec.seed, group_keys=spec.group_keys,
        test_ratio=spec.test_ratio, dev_ratio=spec.dev_ratio,
        stratify_by=spec.stratify_by, holdout_models=tuple(models),
        description=spec.description,
    )


if __name__ == "__main__":
    raise SystemExit(main())
