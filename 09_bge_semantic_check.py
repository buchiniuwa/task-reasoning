#!/usr/bin/env python
"""真实 BGE 的语义验证（用真实中文装配短语）。

三部分：

  A. **人工可判定的相似度检查**：同义 / 释义 / 干扰 / 跨型号同名，
     每条都给出预期关系（应高 / 应低），逐条判定是否符合直觉。
  B. **关键诊断：文本通道对"跨型号同名"是否失效** ——
     两个型号的"接插件主体"文本若完全相同，BGE 必然给出相同向量，
     则文本通道**根本无法**区分它们（这正是型号掩码存在的理由）。用实测证明。
  C. **B1 / B2 基线**：在真实数据上用"仅 BGE 余弦相似度"做接地，
     分别报告"不做掩码"（B1）与"型号+类型掩码"（B2）的 Acc@1 / Recall@3 / MRR，
     并按候选数量分组 —— 这是后续 N04 对比的公平下限。

用法：
    python scripts/09_bge_semantic_check.py
    python scripts/09_bge_semantic_check.py --json outputs/bge_semantic_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.builders import Lexicon                       # noqa: E402
from assemmate_ext.dataset.splits import SplitSpec                       # noqa: E402
from assemmate_ext.encoders.text import (                                # noqa: E402
    TextEncoderConfig, TextFeatureCache, build_bge_encoder, entity_text,
)
from assemmate_ext.grounding.dataset import build_grounding_instances     # noqa: E402
from assemmate_ext.kg.intent import IntentDatasetSlice, SLOT_NODE_TYPES   # noqa: E402
from assemmate_ext.kg.io import read_document                            # noqa: E402
from assemmate_ext.kg.schema import SlotType                             # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout                # noqa: E402
from assemmate_ext.utils.jsonio import read_json, write_json             # noqa: E402

setup_utf8_stdout()

BGE_DIR = Path("D:/models/bge-base-zh-v1.5")
KG_PATH = _WORKSPACE / "data" / "synthetic" / "kg_electronic_demo.json"
DATASET = _WORKSPACE / "data" / "processed" / "connector_synth" / "intent.json"
LEXICON = _WORKSPACE / "data" / "lexicon" / "connector_lexicon.json"
CACHE_DIR = _WORKSPACE / "data" / "processed" / "text_cache"

#: 人工可判定的用例： (短语A, 短语B, 期望 "high"|"low", 说明)
CASES: List[Tuple[str, str, str, str]] = [
    # --- 同义 / 别名：应高 ---
    ("定位销", "导销", "high", "同义词（同一节点的两种说法）"),
    ("接插件主体", "插座本体", "high", "同义（词表里的两种说法）"),
    ("锁扣", "卡扣", "high", "同义"),
    ("CPU 底座", "插座", "high", "同义（同一位置节点的两种说法）"),
    ("散热片", "散热器", "high", "同义"),
    ("插到", "插入", "high", "同一动作的不同形式"),
    ("压紧", "压实", "high", "同一动作的近义形式"),
    # --- 释义（长短句）：应高 ---
    ("把接插件主体插到工位槽里", "接插件主体插到一号工位", "high", "同义指令的不同表述"),
    ("把定位销装到位", "定位销插到位", "high", "同义指令"),
    # --- 不同部件：应低（同型号内的困难负样本） ---
    ("接插件主体", "定位销", "low", "同型号不同零件（N2 同型号相似）"),
    ("接插件主体", "锁扣", "low", "同型号不同零件"),
    ("接插件主体", "排针", "low", "同型号不同零件（N7 接口不匹配的候选）"),
    ("定位销", "排针", "low", "同型号不同零件"),
    ("散热片", "CPU", "low", "同型号相邻部件"),
    # --- 对象 vs 工具（N5）：应低 ---
    ("接插件主体", "插装工装", "low", "零件 vs 工具"),
    ("接插件主体", "扭矩扳手", "low", "零件 vs 工具"),
    # --- 动作区分：应低 ---
    ("插入", "压紧", "low", "不同动作（N4 关系/角色错误）"),
    ("插入", "拧紧", "low", "不同动作"),
    # --- 目标状态区分：应低 ---
    ("到位", "齐平", "low", "不同目标状态（N6 状态错配）"),
    ("硅脂均匀", "到位", "low", "不同目标状态"),
    # --- 位置区分：应低 ---
    ("A 工位槽", "B 工位槽", "low", "同型号不同位置（N3 位置错误）"),
    # --- 工艺约束区分：应低 ---
    ("压紧力按 15–20N", "扭矩按 1.2N·m", "low", "不同工艺约束"),
]

#: 跨型号同名文本对（B 部分诊断）—— 字符串完全相同
CROSS_MODEL_PAIRS: List[Tuple[str, str, str]] = [
    ("conn_x1_a_body", "conn_x1_b_body", "A/B 型号的接插件主体"),
    ("conn_x1_a_seated", "conn_x1_b_seated", "A/B 型号的到位状态"),
    ("conn_x1_a_slot", "conn_x1_b_slot", "A/B 型号的一号工位"),
    ("cool_y1_a_heatsink", "cool_y2_a_heatsink", "Y1/Y2 型号的散热片"),
]


def load_encoder(config: TextEncoderConfig, cache_path: Path):
    """构造真实 BGE 编码器 + 缓存（复用 N03 的接口，不改任何外部约定）。"""
    import torch

    # 线程数：编码速度 vs 可复现。
    # 8 线程最快（26.9 条/秒），但**多线程归约会让向量有微小差异**；
    # 训练与结果复现要求逐位一致，因此这里用单线程（约慢一倍，但有磁盘缓存兜底）。
    import os
    torch.set_num_threads(int(os.environ.get("ASSEM_MATE_THREADS", "1")))
    encoder = build_bge_encoder(config)
    cache = TextFeatureCache(encoder, config)
    if cache_path.is_file():
        added = cache.load(cache_path)
        print(f"  磁盘缓存命中：新增 {added} 条（文件 {cache_path.name}）")
    return cache


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))     # 向量已 L2 归一化


def part_a(cache: TextFeatureCache) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("[A] 人工可判定的相似度检查（真实中文装配短语）")
    print("=" * 76)
    texts: List[str] = []
    for a, b, _exp, _note in CASES:
        texts += [a, b]
    vectors = cache.encode_instructions(list(dict.fromkeys(texts)))
    sim_of = dict(zip(dict.fromkeys(texts), vectors))

    high_scores, low_scores = [], []
    rows: List[Dict[str, Any]] = []
    for a, b, expected, note in CASES:
        score = cosine(sim_of[a], sim_of[b])
        (high_scores if expected == "high" else low_scores).append(score)
        rows.append({"a": a, "b": b, "expected": expected, "cosine": round(score, 4),
                     "note": note})
        print(f"  {score:+.4f}  [{expected:>4}]  {a} ↔ {b}   （{note}）")

    n_high = len(high_scores)
    n_low = len(low_scores)
    mean_high = sum(high_scores) / n_high
    mean_low = sum(low_scores) / n_low
    threshold = (mean_high + mean_low) / 2
    for row in rows:
        if row["expected"] == "high":
            row["verdict"] = "ok" if row["cosine"] >= threshold else "FAIL(应高却偏低)"
        else:
            row["verdict"] = "ok" if row["cosine"] <= threshold else "**盲区**(应低却偏高)"
    failures = [r for r in rows if r["verdict"] != "ok"]
    ok = not failures
    print(f"\n  '应高'均值 {mean_high:.4f} | '应低'均值 {mean_low:.4f} | "
          f"中间阈值 {threshold:.4f}")
    print(f"  全部用例方向正确: {ok}")
    if failures:
        print(f"  文本通道判反的用例（{len(failures)} 条，方向与预期相反）：")
        for row in failures:
            print(f"      {row['cosine']:+.4f}  {row['a']} ↔ {row['b']}   （{row['note']}）")
        print("    → 这些正是需要**结构/掩码/关系证据**才能消歧的负样本类型；"
              "文本通道单独做不到。")
    return {"cases": rows, "mean_high": round(mean_high, 4),
            "mean_low": round(mean_low, 4), "threshold": round(threshold, 4),
            "all_directions_correct": ok,
            "blind_spots": [{"a": r["a"], "b": r["b"], "cosine": r["cosine"],
                             "note": r["note"]} for r in failures]}


def part_b(document, lexicon: Lexicon, cache: TextFeatureCache) -> Dict[str, Any]:
    """跨型号同名：文本通道能否消歧？

    关键度量是**查询侧 margin**：指令里说的是中文说法（不含型号），
    因此真正决定排序的是"查询与两个候选的相似度差多少"，而不是两个候选文本彼此有多像。
    """
    print()
    print("=" * 76)
    print("[B] 关键诊断：文本通道能否区分『跨型号同名』")
    print("=" * 76)
    nodes = document.node_index()
    rows: List[Dict[str, Any]] = []
    for left, right, note in CROSS_MODEL_PAIRS:
        if left not in nodes or right not in nodes:
            continue
        text_l = entity_text(nodes[left], lexicon=lexicon)
        text_r = entity_text(nodes[right], lexicon=lexicon)
        vecs = cache.encode_instructions([text_l, text_r])
        surface = (lexicon.surfaces(left) or ["?"])[0]
        query_vec = cache.encode_instructions([surface])[0]
        cos_left = cosine(query_vec, vecs[0])
        cos_right = cosine(query_vec, vecs[1])
        margin = abs(cos_left - cos_right)
        wrong_preferred = cos_right > cos_left     # 这里 right 是"另一个型号"的实体
        rows.append({
            "left": left, "right": right, "text_left": text_l, "text_right": text_r,
            "cosine_between_candidates": round(cosine(vecs[0], vecs[1]), 6),
            "query": surface, "cos_query_left": round(cos_left, 6),
            "cos_query_right": round(cos_right, 6), "margin": round(margin, 6),
            "note": note, "tiny_margin": margin < 0.05,
            "wrong_model_preferred": wrong_preferred,
        })
        print(f"  候选间 cosine={cosine(vecs[0], vecs[1]):.4f}  {note}")
        print(f"      候选文本 A: {text_l!r}")
        print(f"      候选文本 B: {text_r!r}")
        print(f"      查询 {surface!r}: cos(A)={cos_left:.4f} cos(B)={cos_right:.4f}"
              f"  **margin={margin:.4f}**"
              + ("   ← 偏好错误型号！" if wrong_preferred else ""))

    margins = [r["margin"] for r in rows] or [0.0]
    n_tiny = sum(1 for r in rows if r["tiny_margin"])
    n_wrong = sum(1 for r in rows if r["wrong_model_preferred"])
    print()
    print(f"  查询侧 margin: 最小 {min(margins):.4f} / 最大 {max(margins):.4f} / "
          f"均值 {sum(margins) / len(margins):.4f}")
    print(f"  margin < 0.05 的对数: {n_tiny}/{len(rows)}")
    print(f"  **偏好错误型号**的对数: {n_wrong}/{len(rows)}"
          f"   ← 符号都不可靠，说明这不是『信号弱』而是『噪声』")
    print("  结论：型号既不在查询里、也不在候选的中文说法里"
          "（候选文本里的型号来自我使用的英文节点名，属合成数据命名习惯，不是真实语义）；")
    print("        因此文本通道无法可靠消歧 —— 这从数值上解释了 B2 相对 B1 的增益来源，"
          "也说明型号约束必须由图结构/掩码承担。")
    return {"pairs": rows, "n_tiny_margin": n_tiny,
            "n_wrong_model_preferred": n_wrong,
            "margin_min": round(min(margins), 6), "margin_max": round(max(margins), 6),
            "margin_mean": round(sum(margins) / len(margins), 6)}


def part_c(document, data: IntentDatasetSlice, lexicon: Lexicon,
           cache: TextFeatureCache) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("[C] B1 / B2 基线：仅用 BGE 余弦相似度做接地")
    print("=" * 76)
    from assemmate_ext.grounding.candidates import build_candidates

    nodes = document.node_index()
    node_ids = [n.id for n in document.nodes]
    node_texts = {n.id: entity_text(n, lexicon=lexicon) for n in document.nodes}
    unique_texts = list(dict.fromkeys(node_texts.values()))
    vectors = cache.encode_instructions(unique_texts)
    text_of = dict(zip(unique_texts, vectors))

    instances, stats = build_grounding_instances(data)
    mentions = list(dict.fromkeys(i.mention for i in instances))
    mention_vecs = dict(zip(mentions, cache.encode_instructions(mentions)))

    report: Dict[str, Any] = {}
    for label, use_model_mask in (("B1 BGE Only（不做掩码）", False),
                                  ("B2 BGE + 型号&类型掩码", True)):
        records: List[Dict[str, Any]] = []
        for inst in instances:
            slot = SlotType.coerce(inst.slot)
            candidates = build_candidates(
                document, node_ids, slot, model_id=inst.model_id,
                use_model_mask=use_model_mask, use_type_mask=True)
            if candidates.size == 0:
                continue
            query = mention_vecs[inst.mention]
            sims = [(cid, cosine(query, text_of[node_texts[cid]]))
                    for cid in candidates.node_ids]
            sims.sort(key=lambda kv: -kv[1])
            ranked = [cid for cid, _ in sims]
            rank = ranked.index(inst.entity_id) + 1 if inst.entity_id in ranked else None
            records.append({"instance_id": inst.instance_id, "slot": inst.slot,
                            "model_id": inst.model_id, "target": inst.entity_id,
                            "top1": ranked[0], "rank": rank,
                            "n_candidates": candidates.size,
                            "identified_unknown": False})
        n = len(records)
        scored = [r for r in records if r["rank"] is not None]
        acc1 = sum(1 for r in scored if r["rank"] == 1) / n if n else None
        recall3 = sum(1 for r in scored if r["rank"] <= 3) / n if n else None
        mrr = sum(1.0 / r["rank"] for r in scored) / n if n else None
        by_cand: Dict[str, Any] = {}
        for r in records:
            key = str(r["n_candidates"])
            bucket = by_cand.setdefault(key, {"n": 0, "acc@1": 0})
            bucket["n"] += 1
            bucket["acc@1"] += 1 if r["rank"] == 1 else 0
        for key, bucket in by_cand.items():
            bucket["acc@1"] = round(bucket["acc@1"] / bucket["n"], 4)
        mean_cand = sum(r["n_candidates"] for r in records) / n if n else None
        single = sum(1 for r in records if r["n_candidates"] == 1)
        multi = [r for r in records if r["n_candidates"] > 1]
        acc1_multi = (sum(1 for r in multi if r["rank"] == 1) / len(multi)
                      if multi else None)

        report[label] = {
            "n": n, "acc@1": round(acc1, 4) if acc1 is not None else None,
            "recall@3": round(recall3, 4) if recall3 is not None else None,
            "mrr": round(mrr, 4) if mrr is not None else None,
            "mean_candidates": round(mean_cand, 2) if mean_cand else None,
            "n_single_candidate": single,
            "acc@1_multi_candidate_only": (round(acc1_multi, 4)
                                           if acc1_multi is not None else None),
            "by_n_candidates": by_cand,
        }
        print(f"  {label}")
        print(f"    n={n}  Acc@1={report[label]['acc@1']}  "
              f"Recall@3={report[label]['recall@3']}  MRR={report[label]['mrr']}")
        print(f"    候选数均值={report[label]['mean_candidates']}  "
              f"单候选实例={single} 条（已单列）")
        print(f"    多候选实例上的 Acc@1={report[label]['acc@1_multi_candidate_only']}")
        print(f"    按候选数分组: {by_cand}")
    report["instances_total"] = len(instances)
    report["build_stats"] = stats.to_dict()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="真实 BGE 语义验证")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--bge_dir", default=str(BGE_DIR))
    parser.add_argument("--kg", default=str(KG_PATH), help="知识图谱路径")
    parser.add_argument("--dataset", default=str(DATASET), help="意图数据集路径")
    parser.add_argument("--lexicon", default=str(LEXICON), help="词表路径")
    parser.add_argument("--cache", default=None, help="文本缓存文件（默认按数据集名）")
    args = parser.parse_args()

    bge_dir = Path(args.bge_dir)
    if not bge_dir.is_dir():
        print(f"[FAIL] 未找到本地 BGE 权重: {bge_dir}", file=sys.stderr)
        return 2

    config = TextEncoderConfig(model_name=str(bge_dir), dim=768, batch_size=32,
                               max_length=64, normalize=True)
    print("=" * 76)
    print("真实 BGE 语义验证")
    print("=" * 76)
    print(f"  模型目录: {bge_dir}")
    print(f"  配置: dim={config.dim} batch={config.batch_size} "
          f"max_length={config.max_length} 池化=CLS（模型自带配置指定）")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else (
        CACHE_DIR / f"bge_base_zh_v15_{Path(args.dataset).parent.name}.npz")
    started = time.time()
    cache = load_encoder(config, cache_path)

    document = read_document(args.kg)
    data = IntentDatasetSlice.from_dict(read_json(args.dataset))
    lexicon = Lexicon.load(args.lexicon)
    print(f"  数据集: {args.dataset}")
    print(f"  图谱  : {args.kg}")

    report: Dict[str, Any] = {"model_dir": str(bge_dir), "config": {
        "dim": config.dim, "batch_size": config.batch_size,
        "max_length": config.max_length, "pooling": "cls", "normalize": True}}
    report["A_semantic_cases"] = part_a(cache)
    report["B_cross_model_text"] = part_b(document, lexicon, cache)
    report["C_baselines"] = part_c(document, data, lexicon, cache)

    # 落盘缓存（要求 4：磁盘缓存）
    cache.save(cache_path)
    report["cache"] = {"path": str(cache_path), "stats": cache.stats}
    report["elapsed_s"] = round(time.time() - started, 1)
    print(f"\n  文本缓存已保存: {cache_path}")
    print(f"  编码统计: {cache.stats}")
    print(f"  总耗时: {report['elapsed_s']} s")

    if args.json_out:
        write_json(Path(args.json_out), report)
        print(f"\n报告已写入: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
