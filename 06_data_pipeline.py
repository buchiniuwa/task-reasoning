#!/usr/bin/env python
"""正式数据流水线入口（开发优先级一）。

子命令：

    lexicon-check   检查指令词表对图谱的覆盖（缺条目会导致样本被拒绝生成）
    build           从 KG + 词表构造意图样本与困难负样本（程序化、可复现）
    validate        跑全套校验 + 数据质量报告
    split           生成四套划分并做防泄漏检查
    review-export   导出可人工审阅的 CSV
    review-import   回导人工审阅结果，输出变更日志与新数据集
    report          只输出数据质量报告
    all             build → validate → split → report

数据组织（对应优先级一·第 1 项）：

    data/raw/         只读原始资料（真实 BOM/SOP 放这里，当前为空）
    data/lexicon/     人工维护的节点表面说法词表
    data/synthetic/   手工构造的合成 KG（格式与语义示例）
    data/provenance/  来源登记表（所有数据必须可追溯到这里）
    data/processed/   流水线产物（可随时重建，不手工编辑）

用法示例：
    python scripts/06_data_pipeline.py all
    python scripts/06_data_pipeline.py build --models CONN_X1_A CONN_X1_B
    python scripts/06_data_pipeline.py split --seeds 42 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.builders import (              # noqa: E402
    GENERATOR, BuildConfig, Lexicon, build_dataset,
)
from assemmate_ext.dataset.provenance import (            # noqa: E402
    ProvenanceRegistry, SourceKind, SourceRecord,
)
from assemmate_ext.dataset.quality import build_quality_report   # noqa: E402
from assemmate_ext.dataset.review import export_for_review, import_review  # noqa: E402
from assemmate_ext.dataset.splits import (                # noqa: E402
    PROTOCOL_SPECS, SplitSpec, assign_split, check_leakage,
)
from assemmate_ext.kg.intent import (                     # noqa: E402
    ALL_SLOTS, IntentDatasetSlice, SLOT_NODE_TYPES,
)
from assemmate_ext.kg.intent_validate import validate_intent_dataset  # noqa: E402
from assemmate_ext.kg.io import read_document, write_document   # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout       # noqa: E402
from assemmate_ext.utils.jsonio import read_json, write_json    # noqa: E402

setup_utf8_stdout()

DEFAULT_KG = _WORKSPACE / "data" / "synthetic" / "kg_electronic_demo.json"
DEFAULT_LEXICON = _WORKSPACE / "data" / "lexicon" / "connector_lexicon.json"
DEFAULT_REGISTRY = _WORKSPACE / "data" / "provenance" / "sources.json"
PROCESSED = _WORKSPACE / "data" / "processed"
OUTPUTS = _WORKSPACE / "outputs"


# ====================================================================== 载入
def _rel(path) -> str:
    """把路径转成相对工作区的 POSIX 形式；不在工作区内（或路径未解析）则返回绝对路径。

    实测踩到过：命令行传相对路径时 `Path("data/x").relative_to(ABS_WORKSPACE)` 会抛
    ValueError（一个相对一个绝对）—— 所以必须先 resolve 再降级。
    """
    try:
        return Path(path).resolve().relative_to(_WORKSPACE).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def _load_lexicon(path: Path) -> Lexicon:
    if not path.is_file():
        raise SystemExit(f"[FAIL] 词表不存在: {path}")
    return Lexicon.load(path)


def _load_documents(paths: List[Path]):
    docs = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"[FAIL] 图谱不存在: {path}")
        docs.append(read_document(path))
    return docs


def _load_registry(path: Path, create: bool = True) -> ProvenanceRegistry:
    if path.is_file():
        return ProvenanceRegistry.load(path)
    if not create:
        return ProvenanceRegistry()
    return ProvenanceRegistry()


def _register_generated(registry: ProvenanceRegistry, result, config: BuildConfig) -> List[str]:
    """把本次生成用到的来源编号登记为 synthetic_template（生成器自证来源）。"""
    added: List[str] = []
    ids = sorted({sid for sample in result.samples for sid in sample.evidence_source_ids})
    for source_id in ids:
        if source_id in registry:
            continue
        registry.add(SourceRecord(
            source_id=source_id, kind=SourceKind.SYNTHETIC_TEMPLATE,
            description=f"由 {config.generator} 程序化构造的合成数据（非真实产品资料）",
            origin=f"scripts/06_data_pipeline.py build --models {' '.join(config.models)}",
            created_by=GENERATOR,
            notes="合成数据；不得作为真实工业验证结果引用",
        ), replace=True)
        added.append(source_id)
    return added


def _ensure_manual_sources(registry: ProvenanceRegistry, documents) -> None:
    """把手写合成 KG 里引用的 SYN-ANN-* 来源补登记（合成、人工撰写）。"""
    refs = {n.evidence.ref for d in documents for n in d.nodes if n.evidence.ref}
    refs |= {t.evidence.ref for d in documents for t in d.triples if t.evidence.ref}
    for ref in sorted(r for r in refs if r):
        if ref in registry:
            continue
        kind = (SourceKind.SYNTHETIC_HANDWRITTEN
                if str(ref).upper().startswith("SYN") else SourceKind.UNKNOWN)
        registry.add(SourceRecord(
            source_id=ref, kind=kind,
            description="手工构造的合成演示知识图谱（非真实产品资料）",
            origin=_rel(DEFAULT_KG),
            created_by="scripts/05_build_synthetic_demo.py",
            notes="合成数据；接口与格式示例",
        ), replace=True)


# ====================================================================== 子命令
def cmd_lexicon_check(args) -> int:
    docs = _load_documents(args.kg)
    lex = _load_lexicon(args.lexicon)
    doc = docs[0]
    needed: set = set()
    for node in doc.nodes:
        for slot, types in SLOT_NODE_TYPES.items():
            if node.node_type in types:
                needed.add(node.id)
    missing = lex.missing(sorted(needed))
    print("=" * 74)
    print("词表覆盖检查")
    print("=" * 74)
    print(f"  词表条目      : {len(lex)}")
    print(f"  图谱中可能需要: {len(needed)} 个节点（按槽位类型筛选）")
    print(f"  缺少中文说法  : {len(missing)} 个")
    for node_id in missing[:20]:
        print(f"      - {node_id}")
    if len(missing) > 20:
        print(f"      ... 另有 {len(missing) - 20} 个")
    print()
    print("说明：缺少词条的节点会导致相关样本被**拒绝生成**（不会自动编词）。")
    return 0 if not missing else 1


def cmd_build(args) -> int:
    docs = _load_documents(args.kg)
    doc = docs[0]
    lex = _load_lexicon(args.lexicon)
    registry = _load_registry(args.registry)
    _ensure_manual_sources(registry, docs)

    config = BuildConfig(
        models=tuple(args.models), family=args.family, seed=args.seed,
        n_paraphrases=args.paraphrases,
        max_negatives_per_slot=args.negatives_per_slot,
        max_ops_per_model=args.max_ops_per_model,
    )
    result = build_dataset(doc, lex, config)
    added = _register_generated(registry, result, config)

    out_dir = PROCESSED / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_slice(name=args.name, models=list(config.models)).to_dict()
    payload["meta"].update({
        "purpose": "流程验证用的合成数据（非正式数据集，不得作为工业验证结果）",
        "kg_source": _rel(args.kg[0]),
        "lexicon": _rel(args.lexicon),
        "config": config.to_dict(),
    })
    write_json(out_dir / "intent.json", payload)
    write_json(out_dir / "build_stats.json",
               {"stats": result.stats,
                "skipped_detail": [s.to_dict() for s in result.skipped]})
    registry.save(args.registry)

    print("=" * 74)
    print(f"数据构建完成 → {out_dir}")
    print("=" * 74)
    print(f"  样本 {result.stats['n_samples']} 条 / 负样本 {result.stats['n_negatives']} 对 / "
          f"分组 {result.stats['n_groups']} 个")
    print(f"  场景分布: {result.stats['by_scenario']}")
    print(f"  负样本类型: {result.stats['by_negative_type']}")
    print(f"  掩码后困难占比: {result.stats['hard_negative_ratio']}")
    print(f"  新增登记来源: {added}")
    if result.stats["skipped"]:
        print(f"  跳过原因: {result.stats['skipped']}")
        print("  （跳过是**设计行为**：词表缺失/图谱结构不支持/自检不通过时宁可不出数据）")
    return 0


def _load_dataset(name: str) -> IntentDatasetSlice:
    path = PROCESSED / name / "intent.json"
    if not path.is_file():
        raise SystemExit(f"[FAIL] 未找到构建产物 {path}，请先运行 build")
    return IntentDatasetSlice.from_dict(read_json(path))


def cmd_validate(args) -> int:
    docs = _load_documents(args.kg)
    data = _load_dataset(args.name)
    lex = _load_lexicon(args.lexicon)
    registry = _load_registry(args.registry, create=False)
    _ensure_manual_sources(registry, docs)

    # 若已生成划分，一并纳入质量报告（G8 需要它才有意义）
    splits_index: Dict[str, Any] = {}
    index_path = OUTPUTS / "splits" / "_index.json"
    if index_path.is_file():
        splits_index = dict(read_json(index_path))
    report = build_quality_report(
        docs, data, lexicon=lex, registry=registry, splits=splits_index or None,
        strict_scale=args.strict_scale)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUTS / f"quality_{args.name}.json", report.to_dict())
    print(report.format_text())
    print(f"报告已写入: {OUTPUTS / f'quality_{args.name}.json'}")
    return 0 if report.ok else 1


def cmd_split(args) -> int:
    data = _load_dataset(args.name)
    out_dir = OUTPUTS / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    index: Dict[str, Dict[str, Any]] = {}
    for protocol, base_spec in PROTOCOL_SPECS.items():
        for seed in args.seeds:
            spec = SplitSpec(
                protocol=base_spec.protocol, seed=seed,
                group_keys=base_spec.group_keys, test_ratio=base_spec.test_ratio,
                dev_ratio=base_spec.dev_ratio, stratify_by=base_spec.stratify_by,
                holdout_models=tuple(args.holdout_models)
                if protocol is base_spec.protocol else (),
                min_test=args.min_test, min_dev=args.min_dev,
                description=base_spec.description,
            )
            result = assign_split(data.samples, spec)
            leakage = check_leakage(result, data)
            tag = f"{protocol.value}_seed{seed}"
            write_json(out_dir / f"{tag}.json",
                       {**result.to_dict(), "leakage": leakage.to_dict()})
            index[tag] = {"ok": leakage.ok, "train": len(result.train),
                          "dev": len(result.dev), "test": len(result.test),
                          "protocol": protocol.value, "seed": seed}
            status = "OK  " if leakage.ok else "FAIL"
            if not leakage.ok:
                failures += 1
            print(f"  [{status}] {tag:<46} train={len(result.train):>4} "
                  f"dev={len(result.dev):>4} test={len(result.test):>4}")
            for issue in leakage.errors:
                print(f"          ! {issue}")
            if result.warnings and args.verbose:
                for warning in result.warnings[:2]:
                    print(f"          · {warning}")
    write_json(out_dir / "_index.json", index)
    print(f"划分结果已写入: {out_dir}")
    print(f"泄漏检查未通过: {failures} 个")
    return 0 if failures == 0 else 1


def cmd_review_export(args) -> int:
    data = _load_dataset(args.name)
    out_dir = PROCESSED / args.name
    written = export_for_review(data, out_dir / "review.csv", out_dir / "review_base.jsonl")
    print(f"已导出待审阅文件: {written}")
    print("审阅流程：填写 decision(accept/revise/reject) 与 reviewer 列，"
          "必要时修改槽位状态/取值，然后运行 review-import 回导。")
    return 0


def cmd_review_import(args) -> int:
    docs = _load_documents(args.kg)
    data = _load_dataset(args.name)
    csv_path = Path(args.csv) if args.csv else PROCESSED / args.name / "review.csv"
    result = import_review(csv_path, data)
    print(result.format_text())
    if not result.ok:
        print("回导未通过，未写入新数据（请修正后重试）。")
        return 1
    out_dir = PROCESSED / args.name
    write_json(out_dir / "intent_reviewed.json",
               IntentDatasetSlice(name=f"{args.name}_reviewed", samples=result.samples,
                                  negatives=data.negatives,
                                  meta={"reviewed": True}).to_dict())
    write_json(out_dir / "review_changelog.json", result.to_dict())
    print(f"已写入: {out_dir / 'intent_reviewed.json'}")
    print(f"变更日志: {out_dir / 'review_changelog.json'}")
    return 0


def cmd_report(args) -> int:
    return cmd_validate(args)


def cmd_all(args) -> int:
    """顺序：词表检查 → 构建 → 划分 → 校验（校验放最后，这样质量报告里含划分结果）。"""
    rc = cmd_lexicon_check(args)
    if rc != 0 and args.strict_lexicon:
        print("[FAIL] 词表覆盖不全，终止（可用 --no-strict-lexicon 忽略）")
        return 1
    rc_build = cmd_build(args)
    if rc_build != 0:
        return rc_build
    print()
    rc_split = cmd_split(args)
    print()
    rc_val = cmd_validate(args)
    return 0 if (rc_val == 0 and rc_split == 0) else 1


# ====================================================================== 入口
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="研究点一数据流水线")
    parser.add_argument("command", choices=[
        "lexicon-check", "build", "validate", "split", "review-export",
        "review-import", "report", "all"])
    parser.add_argument("--kg", type=Path, nargs="+", default=[DEFAULT_KG],
                        help="知识图谱文件（可多个）")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="指令词表")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="来源登记表")
    parser.add_argument("--name", default="connector_synth", help="数据集名称（产物目录名）")
    parser.add_argument("--models", nargs="+",
                        default=["CONN_X1_A", "CONN_X1_B", "COOL_Y1_A", "COOL_Y2_A"],
                        help="参与构建的型号")
    parser.add_argument("--family", default="connector_insertion", help="产品族")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="划分种子（建议正式实验用 5 个）")
    parser.add_argument("--paraphrases", type=int, default=2, help="每样本改写条数")
    parser.add_argument("--negatives-per-slot", type=int, default=2)
    parser.add_argument("--max-ops-per-model", type=int, default=None)
    parser.add_argument("--holdout-models", nargs="*", default=["CONN_X1_B"],
                        help="P2 留出型号（未见型号实验）")
    parser.add_argument("--min-test", type=int, default=40)
    parser.add_argument("--min-dev", type=int, default=20)
    parser.add_argument("--csv", type=Path, default=None, help="review-import 的输入 CSV")
    parser.add_argument("--strict-scale", action="store_true",
                        help="把『正式数据规模』当作错误（正式实验前使用）")
    parser.add_argument("--strict-lexicon", action="store_true", default=False,
                        help="all：词表不全时终止")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    handlers = {
        "lexicon-check": cmd_lexicon_check,
        "build": cmd_build,
        "validate": cmd_validate,
        "split": cmd_split,
        "review-export": cmd_review_export,
        "review-import": cmd_review_import,
        "report": cmd_report,
        "all": cmd_all,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
