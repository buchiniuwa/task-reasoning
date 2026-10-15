#!/usr/bin/env python
"""N01：在演示图谱之上扩充**难度**（不是数量）。

做法：读取 `data/synthetic/kg_electronic_demo.json`，按**显式表**追加节点/边，
生成 `data/synthetic/kg_connector_hard.json` 与配套词表。
不复制模板凑数，每一类新增都对应一个明确的研究难点。

新增的六类难度（对应用户要求）：

| # | 难度类型 | 本次新增 | 为什么文本通道解不了 |
|---|---|---|---|
| 1 | 同型号、同类型、不同实例的相似零件 | `guide_pin` / `guide_pin_2` / `guide_pin_3`（同类型，近名） | "定位销" vs "定位销二号/三号" 文本高度相似 |
| 2 | 同名称、不同安装接口 | `body` 与 `body_spare`（词表说法**完全相同**："接插件主体"） | 查询文本一模一样，cosine=1.0，**只有接口边能区分** |
| 3 | 相同操作对象、不同当前状态 | 新增 op 对，同一对象、不同 `achieves` | 必须靠状态/关系证据 |
| 4 | 语义相近但关系路径不同 | `guide_pin_3` 既是某操作的 `acts_on` 对象、又是另一操作的 `requires_tool` 工具 | 同一节点两种角色，名称无法区分 |
| 5 | 型号正确、类型正确但目标位置错误 | `slot` / `slot_spare`（"A 工位槽" vs "A 工位槽备用"） | 位置名近似 |
| 6 | 只能靠图谱证据唯一确定的隐含目标 | 上述组合：完整槽位下只有**唯一**一个部件满足全部见证边 | 需要联合证据，不能只看名字 |

纪律：全部标记 `synthetic`，来源编号 `SYN-HARD-*`，不触碰任何真实 BOM/SOP。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.kg.io import read_document, write_document      # noqa: E402
from assemmate_ext.kg.validate import validate_document            # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout          # noqa: E402
from assemmate_ext.utils.jsonio import read_json, write_json       # noqa: E402

setup_utf8_stdout()

DEMO_KG = _WORKSPACE / "data" / "synthetic" / "kg_electronic_demo.json"
HARD_KG = _WORKSPACE / "data" / "synthetic" / "kg_connector_hard.json"
DEMO_LEXICON = _WORKSPACE / "data" / "lexicon" / "connector_lexicon.json"
HARD_LEXICON = _WORKSPACE / "data" / "lexicon" / "connector_hard_lexicon.json"

SOURCE = "SYN-HARD-CXA-01"

#: 新增节点：(id, name, node_type, model_id, 备注)
NEW_NODES: List[Dict[str, Any]] = [
    # --- 难度 2：同名不同接口（词表说法与 conn_x1_a_body 完全相同） ---
    {"id": "conn_x1_a_body_spare", "name": "SpareBody_CONN_X1_A",
     "node_type": "component_instance", "model_id": "CONN_X1_A",
     "note": "与 conn_x1_a_body 同名同类型，仅接口边不同"},
    # --- 难度 1：同类型近名实例 ---
    {"id": "conn_x1_a_guide_pin_2", "name": "GuidePin2_CONN_X1_A",
     "node_type": "component_instance", "model_id": "CONN_X1_A",
     "note": "与 conn_x1_a_guide_pin 同类型、名称仅差序号"},
    {"id": "conn_x1_a_guide_pin_3", "name": "GuidePin3_CONN_X1_A",
     "node_type": "component_instance", "model_id": "CONN_X1_A",
     "note": "同类型近名；且在另一操作中充当工具（难度 4）"},
    # --- 难度 5：位置近名 ---
    {"id": "conn_x1_a_slot_spare", "name": "CONN_X1_A_SLOT_A_SPARE",
     "node_type": "location", "model_id": "CONN_X1_A",
     "note": "与 conn_x1_a_slot 名称仅差后缀"},
    # --- 难度 3：同对象不同状态所需的第二个目标状态（flush 已存在，这里补一个工序） ---
    {"id": "conn_x1_a_step3", "name": "CONN_X1_A_STEP3",
     "node_type": "step", "model_id": "CONN_X1_A",
     "note": "承载“同对象不同状态”的操作对"},
]

#: 新增操作与其五条见证边：(op_id, name, action, obj, loc, state, constraint, tool)
NEW_OPS: List[Dict[str, str]] = [
    # 难度 2 + 5：备用主体装到备用工位（与 op1_1 同动作同状态，但对象与位置都不同）
    {"id": "conn_x1_a_op1_2", "action": "insert", "obj": "conn_x1_a_body_spare",
     "loc": "conn_x1_a_slot_spare", "state": "conn_x1_a_seated",
     "constraint": "conn_x1_a_force", "tool": "insertion_tool"},
    # 难度 1 + 3：同对象类型不同实例，且目标状态与其它 pin 不同
    {"id": "conn_x1_a_op1_4", "action": "insert", "obj": "conn_x1_a_guide_pin_2",
     "loc": "conn_x1_a_slot", "state": "conn_x1_a_flush",
     "constraint": "conn_x1_a_force", "tool": "insertion_tool"},
    # 难度 4：pin_3 作为对象
    {"id": "conn_x1_a_op1_5", "action": "insert", "obj": "conn_x1_a_guide_pin_3",
     "loc": "conn_x1_a_slot", "state": "conn_x1_a_seated",
     "constraint": "conn_x1_a_torque", "tool": "insertion_tool"},
    # 难度 3：与 op1_5 同对象，但目标状态/约束/位置不同（同对象不同状态/工艺）
    {"id": "conn_x1_a_op3_1", "action": "press", "obj": "conn_x1_a_guide_pin_3",
     "loc": "conn_x1_a_slot_spare", "state": "conn_x1_a_flush",
     "constraint": "conn_x1_a_torque", "tool": "conn_x1_a_guide_pin_2"},
]

#: 难度 4：把某个**零件**作为工具（requires_tool 指向 component_instance）
PART_AS_TOOL = {"op_id": "conn_x1_a_op1_5", "part_id": "conn_x1_a_guide_pin_2"}

#: 接口兼容（难度 2 的核心：同名不同接口）
COMPATIBLE: List[Dict[str, str]] = [
    {"head": "conn_x1_a_body", "tail": "conn_x1_a_slot",
     "note": "主体只能插 A 工位槽"},
    {"head": "conn_x1_a_body_spare", "tail": "conn_x1_a_slot_spare",
     "note": "备用主体只能插 A 工位槽备用 —— 与主体同名但接口不同"},
    {"head": "conn_x1_a_guide_pin", "tail": "conn_x1_a_slot", "note": "定位销接口"},
    {"head": "conn_x1_a_guide_pin_2", "tail": "conn_x1_a_slot", "note": "定位销二号接口"},
    {"head": "conn_x1_a_guide_pin_3", "tail": "conn_x1_a_slot", "note": "定位销三号接口"},
]

#: 新增词表条目（关键：body_spare 与 body **共用同一说法**）
NEW_LEXICON: List[Dict[str, Any]] = [
    {"node_id": "conn_x1_a_body_spare", "zh": ["接插件主体"],
     "note": "与 conn_x1_a_body **完全同名** —— 文本通道无法区分（难度 2）"},
    {"node_id": "conn_x1_a_guide_pin_2", "zh": ["定位销二号"],
     "note": "与“定位销”近名（难度 1）"},
    {"node_id": "conn_x1_a_guide_pin_3", "zh": ["定位销三号"],
     "note": "与“定位销”近名（难度 1）"},
    {"node_id": "conn_x1_a_slot_spare", "zh": ["A 工位槽备用"],
     "note": "与“A 工位槽”近名（难度 5）"},
]


def build_hard_kg(demo_path: Path, out_path: Path) -> Dict[str, Any]:
    raw = read_json(demo_path)
    doc = copy.deepcopy(raw)

    # ---- 节点 ----
    existing_ids = {n["id"] for n in doc["nodes"]}
    for spec in NEW_NODES:
        if spec["id"] in existing_ids:
            continue
        doc["nodes"].append({
            "id": spec["id"], "name": spec["name"], "node_type": spec["node_type"],
            "model_id": spec["model_id"], "aliases": [],
            "state": {"assembly_state": "unknown"},
            "evidence": {"source": "manual_annotation", "ref": SOURCE,
                         "note": f"合成演示数据（N01 难度扩充）: {spec['note']}"},
            "extra": {"synthetic": True, "difficulty_case": spec["note"]},
        })

    # ---- 操作节点本身也必须建出来（否则三元组的 head 会悬空 —— 实测踩到过）----
    for spec in NEW_OPS:
        if spec["id"] in existing_ids:
            continue
        doc["nodes"].append({
            "id": spec["id"], "name": f"{spec['id'].upper()}",
            "node_type": "operation", "model_id": "CONN_X1_A", "aliases": [],
            "state": {"assembly_state": "unknown"},
            "evidence": {"source": "manual_annotation", "ref": SOURCE,
                         "note": "合成演示数据（N01 难度扩充）：新增操作"},
            "extra": {"synthetic": True},
        })

    seq = max((int(t["id"][1:]) for t in doc["triples"] if t["id"].startswith("t")
               and t["id"][1:].isdigit()), default=0)

    existing_edges = {(t["head"], t["relation"], t["tail"]) for t in doc["triples"]}

    def add_triple(head: str, relation: str, tail: str, model: Optional[str],
                   note: str, extra: Optional[Dict[str, Any]] = None) -> str:
        nonlocal seq
        if (head, relation, tail) in existing_edges:
            # 幂等：演示图谱里可能已经有同一条边（实测踩到过 W009 重复三元组）
            return ""
        seq += 1
        tid = f"t{seq:03d}"
        existing_edges.add((head, relation, tail))
        doc["triples"].append({
            "id": tid, "head": head, "relation": relation, "tail": tail,
            "model_id": model,
            "evidence": {"source": "manual_annotation", "ref": SOURCE,
                         "note": note},
            "state": {"assembly_state": "unknown"},
            "extra": {"synthetic": True, **(extra or {})},
        })
        return tid

    model = "CONN_X1_A"

    # ---- 新节点挂到型号上（instance_of / has_component）----
    type_of = {"conn_x1_a_body_spare": "connector_body",
               "conn_x1_a_guide_pin_2": "guide_pin_type",
               "conn_x1_a_guide_pin_3": "guide_pin_type"}
    for node_id, type_id in type_of.items():
        add_triple(node_id, "instance_of", type_id, model, "难度扩充：部件实例归属其类型")
        add_triple(f"conn_x1_a_model", "has_component", node_id, model,
                   "难度扩充：型号包含该部件")

    # ---- 新工序与操作 ----
    add_triple("conn_x1_a_model", "has_step", "conn_x1_a_step3", model,
               "难度扩充：第三个工序")
    step_of = {"conn_x1_a_op1_2": "conn_x1_a_step1", "conn_x1_a_op1_4": "conn_x1_a_step1",
               "conn_x1_a_op1_5": "conn_x1_a_step1", "conn_x1_a_op3_1": "conn_x1_a_step3"}
    for spec in NEW_OPS:
        op_id = spec["id"]
        add_triple(step_of[op_id], "has_operation", op_id, model,
                   f"难度扩充：{op_id} 归属工序")
        for relation, key in (("has_action", "action"), ("acts_on", "obj"),
                              ("acts_at", "loc"), ("achieves", "state"),
                              ("constrained_by", "constraint"), ("requires_tool", "tool")):
            add_triple(op_id, relation, spec[key], model,
                       f"难度扩充：{op_id} 的 {relation} 见证边")

    # ---- 难度 4：把零件当工具（同一节点两种角色）----
    add_triple(PART_AS_TOOL["op_id"], "requires_tool", PART_AS_TOOL["part_id"], model,
               "难度扩充：**零件被当作工具使用** —— 同一节点在不同操作中角色不同",
               extra={"difficulty": "part_as_tool"})

    # ---- 难度 2：接口兼容边（区分同名不同接口）----
    for spec in COMPATIBLE:
        add_triple(spec["head"], "compatible_with", spec["tail"], model, spec["note"],
                   extra={"difficulty": "interface"})

    # ---- 型号声明补充 ----
    for entry in doc.get("models", []):
        if entry.get("model_id") == "CONN_X1_A":
            for extra_type in ("guide_pin_type",):
                if extra_type not in entry.get("component_types", []):
                    entry.setdefault("component_types", []).append(extra_type)

    # ---- 元信息 ----
    doc["doc_id"] = "kg_connector_hard"
    doc["meta"] = dict(doc.get("meta", {}))
    doc["meta"].update({
        "synthetic": True,
        "derived_from": demo_path.name,
        "purpose": (
            "N01 难度扩充版：在演示图谱之上追加六类困难候选，"
            "用于让『掩码后候选集合』包含文本不可区分的项。"
            "仍为合成数据，不得作为真实工业资料引用。"
        ),
        "added_nodes": [n["id"] for n in NEW_NODES],
        "added_operations": [o["id"] for o in NEW_OPS],
        "difficulty_cases": [
            "同型号同类型近名实例（定位销 / 二号 / 三号）",
            "同名称不同接口（接插件主体 / 接插件主体）",
            "相同操作对象不同目标状态（op1_5 与 op3_1）",
            "同一节点两种角色（guide_pin_2 既是对象又是工具）",
            "位置近名（A 工位槽 / A 工位槽备用）",
            "只能靠联合证据唯一确定（完整槽位下仅一个部件满足全部见证边）",
        ],
        "note": "来源编号 SYN-HARD-CXA-01；未引入任何真实 BOM/SOP 内容",
    })

    write_json(out_path, doc)
    return doc


def build_hard_lexicon(demo_lexicon: Path, out_path: Path,
                       doc: Dict[str, Any]) -> Dict[str, Any]:
    raw = read_json(demo_lexicon)
    entries = {e["node_id"]: e for e in raw.get("entries", [])}
    for spec in NEW_LEXICON:
        entries[spec["node_id"]] = spec
    payload = {
        "meta": {
            "synthetic": True,
            "derived_from": demo_lexicon.name,
            "purpose": ("N01 难度扩充版词表：新增条目刻意与既有条目**同名或近名**，"
                        "使文本通道不足以区分候选。"),
            "note": "仍然只允许用这里的说法生成指令；缺条目会导致样本被拒绝生成",
        },
        "entries": [entries[k] for k in sorted(entries)],
    }
    write_json(out_path, payload)

    # 名称冲突检查（同名是**有意**的，但要显式报告，避免被误认为脏数据）
    surface_to_ids: Dict[str, List[str]] = {}
    for entry in payload["entries"]:
        for surface in entry.get("zh", []):
            surface_to_ids.setdefault(surface, []).append(entry["node_id"])
    duplicates = {s: ids for s, ids in surface_to_ids.items() if len(ids) > 1}
    return {"n_entries": len(payload["entries"]), "duplicate_surfaces": duplicates}


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 N01 难度扩充图谱")
    parser.add_argument("--demo_kg", default=str(DEMO_KG))
    parser.add_argument("--out_kg", default=str(HARD_KG))
    parser.add_argument("--out_lexicon", default=str(HARD_LEXICON))
    args = parser.parse_args()

    doc = build_hard_kg(Path(args.demo_kg), Path(args.out_kg))
    document = read_document(args.out_kg)
    report = validate_document(document)

    print("=" * 74)
    print("N01 难度扩充图谱")
    print("=" * 74)
    print(f"  节点 {len(document.nodes)}  关系 {len(document.relations)}  "
          f"三元组 {len(document.triples)}  型号 {document.available_models()}")
    print(report.format_text())

    lex_info = build_hard_lexicon(Path(DEMO_LEXICON), Path(args.out_lexicon), doc)
    print(f"\n  词表条目 {lex_info['n_entries']}")
    if lex_info["duplicate_surfaces"]:
        print("  **刻意制造的同名说法**（文本通道无法区分的根源）：")
        for surface, ids in sorted(lex_info["duplicate_surfaces"].items()):
            print(f"      {surface!r} → {ids}")
    print(f"\n  已写入: {args.out_kg}")
    print(f"  已写入: {args.out_lexicon}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
