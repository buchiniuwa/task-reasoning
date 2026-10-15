#!/usr/bin/env python
"""统一 Schema 的校验演示。

做两件事：
  1. 校验手工构造的合成示例（data/synthetic/schema_demo.json），应无错误；
  2. **故意**篡改若干字段，演示校验器确实能抓到问题（E005/E010/E013/E017 等）。
     篡改只发生在内存里，绝不写回文件。

注意：合成示例仅用于验证接口，不是真实数据，也不参与任何训练或评测。

用法：
    python scripts/03_validate_schema_demo.py
    python scripts/03_validate_schema_demo.py --file data/synthetic/schema_demo.json
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.kg.io import read_document                 # noqa: E402
from assemmate_ext.kg.validate import validate_document       # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout     # noqa: E402
from assemmate_ext.utils.jsonio import read_json              # noqa: E402

setup_utf8_stdout()

DEFAULT_FILE = _WORKSPACE / "data" / "synthetic" / "schema_demo.json"


def _tamper_cases(raw: dict) -> list:
    """构造若干"被破坏的文档"，每个只改一处，用来证明校验规则有效。"""

    def case(name, desc, mutate):
        broken = copy.deepcopy(raw)
        mutate(broken)
        return name, desc, broken

    def break_head(b):
        b["triples"][0]["head"] = "不存在的节点id"

    def break_cross_model(b):
        # 把跨型号兼容关系换成不允许跨型号的 acts_on
        for t in b["triples"]:
            if t["relation"] == "compatible_with":
                t["relation"] = "acts_on"

    def break_direction(b):
        for t in b["triples"]:
            if t["relation"] == "instance_of":
                t["head"], t["tail"] = t["tail"], t["head"]
                break

    def break_state(b):
        b["nodes"][1]["state"]["assembly_state"] = "done"   # 非法枚举值

    def break_confidence(b):
        b["triples"][0]["confidence"] = 1.7                 # 越界

    def break_version(b):
        b["schema_version"] = "9.9.9"

    return [
        case("E005", "三元组 head 指向不存在的节点", break_head),
        case("E010", "跨型号关系未声明 cross_model", break_cross_model),
        case("E013", "instance_of 方向颠倒", break_direction),
        case("E017", "状态值非法（done）", break_state),
        case("E012", "confidence 越界（1.7）", break_confidence),
        case("W008", "schema_version 不一致", break_version),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="统一 Schema 校验演示")
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="要校验的图谱文档")
    parser.add_argument("--expect-errors", action="store_true", default=True,
                        help="同时演示错误检出（默认开启）")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"[FAIL] 文件不存在: {path}", file=sys.stderr)
        return 1

    print("=" * 74)
    print("第 1 部分：校验合成示例（期望：无错误）")
    print("=" * 74)
    doc = read_document(path)
    report = validate_document(doc)
    print(report.format_text())

    print()
    print(f"文档概览: doc_id={doc.doc_id}  型号={doc.available_models()}  "
          f"节点={len(doc.nodes)}  关系={len(doc.relations)}  三元组={len(doc.triples)}")

    print()
    print("=" * 74)
    print("第 2 部分：故意破坏字段，验证校验器能抓到（只在内存中篡改，不写回文件）")
    print("=" * 74)
    raw = read_json(path)
    from assemmate_ext.kg.io import document_from_dict

    all_caught = True
    for expected_code, desc, broken_raw in _tamper_cases(raw):
        try:
            broken_doc = document_from_dict(broken_raw, source="<内存篡改>")
        except Exception as exc:
            print(f"  破坏方式 {expected_code:<5} {desc}\n      -> 解析期即被拦截: {type(exc).__name__}: {exc}")
            continue
        vr = validate_document(broken_doc)
        caught = vr.has(expected_code)
        all_caught = all_caught and caught
        sample = next((i for i in vr.issues if i.code == expected_code), None)
        mark = "已捕获" if caught else "**未捕获**"
        print(f"  {expected_code:<5} {desc}\n      -> {mark}"
              + (f": {sample.message}" if sample else ""))

    print()
    print(f"合成示例校验通过: {report.ok}（错误 {len(report.errors)} / 警告 {len(report.warnings)}）")
    print(f"全部故意破坏均被捕获: {all_caught}")
    return 0 if (report.ok and all_caught) else 1


if __name__ == "__main__":
    raise SystemExit(main())
