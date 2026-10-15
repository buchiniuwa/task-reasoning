#!/usr/bin/env python
"""第四阶段：关系反事实诊断（**区分"分数变了"与"选对了"**）。

用户明确要求分开报告的两件事：

1. **关系边改变是否导致分数变化** —— 敏感性；
2. **关系边改变后模型能否正确选择实体** —— 正确性。

一个模型可以"分数对关系边很敏感"，但**敏感的方向是错的**（换条边就选错）。
只报其中一个都会造成误读，所以本脚本把两类指标并列打印。

诊断项：

| 指标 | 定义 |
|---|---|
| 成对反事实正确率 | 同一对的两个变体**都**答对的比例 |
| 正确切换率 | 在"答案应当翻转"的变体上，模型确实翻转的比例 |
| 正确实体的 score margin | gold 分数 − 最高错误候选分数 |
| 关键关系边干预前后的分数变化 | 移除 gold 的 `decisive_triples` 后，gold 候选分数的变化 |
| 证据路径匹配正确率 | top-1 候选是否拥有 ≥1 条证据路径；路径计数排序的命中率 |
| conflict / ambiguous / unknown 分类 | 逐类 P/R/F1 |

用法：
    python scripts/17_counterfactual_diagnostics.py --epochs 60
    python scripts/17_counterfactual_diagnostics.py --systems B2 B2+ C D
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

import torch                                                            # noqa: E402
import torch.nn.functional as F                                          # noqa: E402

from assemmate_ext.encoders.compgcn.model import CompGCNConfig           # noqa: E402
from assemmate_ext.grounding.baselines import (                          # noqa: E402
    deterministic_relational_query_with_trust,
)
from assemmate_ext.grounding.dataset import GroundingInstance            # noqa: E402
from assemmate_ext.grounding.evidence_scorer import (                    # noqa: E402
    EvidenceScorer, EvidenceScorerConfig,
)
from assemmate_ext.grounding.inference import neural_predictions        # noqa: E402
from assemmate_ext.grounding.metrics import (                            # noqa: E402
    Prediction, evaluate, format_report, prediction_from_query_result,
)
from assemmate_ext.grounding.ocrm import OCRM, OCRMConfig                # noqa: E402
from assemmate_ext.grounding.objective import (                         # noqa: E402
    class_weights, compute_losses,
)
from assemmate_ext.grounding.ocrm_data import STATUS_ORDER, OcrMBatch    # noqa: E402
from assemmate_ext.grounding.operation_context import (                  # noqa: E402
    build_operation_context, deterministic_relational_query,
)
from assemmate_ext.grounding.runtime import (                            # noqa: E402
    DEFAULT_BGE_DIR, GroundingRuntime, build_runtime, set_determinism,
)
from assemmate_ext.utils.console import setup_utf8_stdout                # noqa: E402
from assemmate_ext.utils.jsonio import write_json                        # noqa: E402

setup_utf8_stdout()

NON_UNIQUE = ("ambiguous", "unknown", "conflict")


# ====================================================================== 上下文
def _context_of(instance: GroundingInstance, document: Any):
    """用实例**已观测到的**线索重建上下文（与其它脚本同一份输入）。"""
    from assemmate_ext.grounding.ocrm_data import _hints_of

    return build_operation_context(
        document, instance.mention_text, instance.slot, instance.candidate_ids,
        instruction=instance.full_instruction, model_id=instance.model_id,
        completed_ops=instance.completed_ops, current_state=instance.current_state,
        hints=_hints_of(instance), operation_candidates=instance.operation_candidates)


# ====================================================================== 神经网络推理
def train_model(model: torch.nn.Module, batch: OcrMBatch, epochs: int, lr: float,
                seed: int, lambda_status: float = 0.1,
                lambda_source: float = 0.1) -> Dict[str, Any]:
    """与 `scripts/16_*` 完全相同的确定性训练（共用同一套目标函数语义）。"""
    torch.manual_seed(seed)
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    weights = class_weights(batch)
    history = []
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad(set_to_none=True)
        out = model(batch)
        losses = compute_losses(out, batch, weights,
                                lambda_status=lambda_status,
                                lambda_source=lambda_source)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if epoch == 1 or epoch % max(epochs // 4, 1) == 0 or epoch == epochs:
            history.append({"epoch": epoch,
                            "loss": round(float(losses["total"].detach()), 5),
                            "align": round(float(losses["align"].detach()), 5),
                            "status": round(float(losses["status"].detach()), 5)})
    return {"history": history, "class_weights": [round(float(w), 4) for w in weights]}


# ====================================================================== 干预
def _strip_decisive(document: Any, triples: Sequence[str]) -> Any:
    """从可见图谱里删掉给定的 `head|relation|tail` 三元组（= 关键关系边干预）。"""
    keys = set()
    for item in triples:
        parts = str(item).split("|")
        if len(parts) == 3:
            keys.add(tuple(parts))
    kept = [t for t in document.triples if (t.head, t.relation, t.tail) not in keys]
    return replace(document, triples=kept)


def _reobserve(instance: GroundingInstance, document: Any) -> GroundingInstance:
    """在**改动后的可见图谱**上重新观测：线索 / 候选操作 / 证据特征 / 证据路径全部重算。

    ⚠️ 只改文档而不重算是不够的：`GroundingInstance` 里的证据是**预计算**的，
    若沿用旧证据，干预只影响了 CompGCN 的边表，而"证据特征"还写着老样子，
    测出来的分数变化会严重低估真实敏感性（第一次实现就踩了这个坑，实测 Δ 恒为 0）。
    """
    from assemmate_ext.grounding.ocrm_data import _hints_of
    from assemmate_ext.grounding.operation_context import (
        evidence_features, relation_evidence_paths, source_features,
    )

    hints = _hints_of(instance)
    context = build_operation_context(
        document, instance.mention_text, instance.slot, instance.candidate_ids,
        instruction=instance.full_instruction, model_id=instance.model_id,
        completed_ops=instance.completed_ops, current_state=instance.current_state,
        hints=hints, operation_candidates=instance.operation_candidates)
    return replace(
        instance,
        candidate_evidence={cid: evidence_features(document, cid, context)
                            for cid in instance.candidate_ids},
        candidate_relation_evidence={cid: relation_evidence_paths(document, cid, context)
                                     for cid in instance.candidate_ids},
        candidate_source_evidence={cid: source_features(document, cid, context)
                                   for cid in instance.candidate_ids},
        operation_candidates=list(context.op_ids))


def intervention_deltas(model: torch.nn.Module, runtime: GroundingRuntime,
                        system: str) -> Dict[str, Any]:
    """**逐条移除关键关系边 + 重新观测**，看 gold 候选的分数掉多少、名次是否翻盘。

    这是"关系边改变导致分数变化"的直接测量；与"能否选对"**分开报告**。
    """
    rows: List[Dict[str, Any]] = []
    model.eval()
    for instance in runtime.instances:
        gold = instance.gold
        if gold is None or gold.status != "grounded" or not gold.decisive_triples:
            continue
        if gold.entity_id not in instance.candidate_ids:
            continue
        document = runtime.documents[instance.visible_graph_id]
        with torch.no_grad():
            before = model(runtime.batch([instance]))["scores"]
        stripped = _strip_decisive(document, gold.decisive_triples)
        intervened = _reobserve(instance, stripped)
        original = runtime.documents
        runtime.documents = dict(original)
        runtime.documents[instance.visible_graph_id] = stripped
        try:
            with torch.no_grad():
                after = model(runtime.batch([intervened]))["scores"]
        finally:
            runtime.documents = original
        index = instance.candidate_ids.index(gold.entity_id)
        delta = float(before[0, index] - after[0, index])
        flipped = bool(int(before[0].argmax()) != int(after[0].argmax()))
        rows.append({"instance_id": instance.instance_id,
                     "score_before": round(float(before[0, index]), 6),
                     "score_after": round(float(after[0, index]), 6),
                     "delta": round(delta, 6), "flipped": flipped,
                     "n_decisive_triples": len(gold.decisive_triples)})
    deltas = [row["delta"] for row in rows]
    flips = sum(1 for row in rows if row["flipped"])
    return {
        "system": system,
        "n_interventions": len(rows),
        "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "min_delta": round(min(deltas), 6) if deltas else None,
        "n_positive_delta": sum(1 for d in deltas if d > 1e-4),
        "n_flipped": flips,
        "flip_rate": round(flips / len(rows), 4) if rows else None,
        "rows": rows,
    }


# ====================================================================== 成对诊断
def pair_diagnostics(instances: Sequence[GroundingInstance],
                     predictions: Sequence[Prediction]) -> Dict[str, Any]:
    """成对反事实正确率 / 正确切换率 / 正确实体的 margin / 证据路径匹配。"""
    by_instance = {p.instance_id: p for p in predictions}
    by_id = {i.instance_id: i for i in instances}
    pairs: Dict[str, List[GroundingInstance]] = defaultdict(list)
    for instance in instances:
        if instance.pair_id:
            pairs[instance.pair_id].append(instance)

    n_pairs = n_both_correct = 0
    switch_examples = switch_ok = 0
    margins: List[float] = []
    evidence_top1_ok = evidence_top1_total = 0
    path_rank_ok = path_rank_total = 0
    rows: List[Dict[str, Any]] = []

    for pair_id, members in sorted(pairs.items()):
        members = sorted(members, key=lambda i: i.variant_id or "")
        base = next((m for m in members if m.variant_id == "base"), None)
        counter = next((m for m in members if m.variant_id == "counterfactual"), None)
        if base is None or counter is None:
            continue
        pb, pc = by_instance.get(base.instance_id), by_instance.get(counter.instance_id)
        if pb is None or pc is None:
            continue
        n_pairs += 1
        ok_b = _prediction_correct(base, pb)
        ok_c = _prediction_correct(counter, pc)
        if ok_b and ok_c:
            n_both_correct += 1

        # 正确切换：base 与 counterfactual 的 gold 实体不同（或状态不同）时，
        # 模型的两个输出也必须不同，并且各自正确
        if _gold_key(base) != _gold_key(counter):
            switch_examples += 1
            if ok_b and ok_c and _pred_key(pb) != _pred_key(pc):
                switch_ok += 1

        for instance, prediction in ((base, pb), (counter, pc)):
            margin = _margin(instance, prediction)
            if margin is not None:
                margins.append(margin)
            total = sum(len(v) for v in
                        (instance.candidate_relation_evidence or {}).values())
            if total:
                evidence_top1_total += 1
                if prediction.entity and (instance.candidate_relation_evidence.get(
                        prediction.entity) or []):
                    evidence_top1_ok += 1
                path_rank_total += 1
                ranked = sorted(instance.candidate_ids,
                                key=lambda c: -(len(instance.candidate_relation_evidence
                                                    .get(c) or [])), )
                if instance.gold and instance.gold.entity_id and \
                        ranked and ranked[0] == instance.gold.entity_id:
                    path_rank_ok += 1
        rows.append({"pair_id": pair_id,
                     "base_gold": _gold_key(base), "counter_gold": _gold_key(counter),
                     "base_pred": _pred_key(pb), "counter_pred": _pred_key(pc),
                     "both_correct": ok_b and ok_c})

    def rate(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    return {
        "n_pairs": n_pairs,
        "pairwise_accuracy": rate(n_both_correct, n_pairs),
        "n_switchable_pairs": switch_examples,
        "correct_switch_rate": rate(switch_ok, switch_examples),
        "margin_mean": round(sum(margins) / len(margins), 4) if margins else None,
        "margin_min": round(min(margins), 4) if margins else None,
        "n_negative_margin": sum(1 for m in margins if m < 0),
        "evidence_top1_rate": rate(evidence_top1_ok, evidence_top1_total),
        "path_rank_accuracy": rate(path_rank_ok, path_rank_total),
        "rows": rows,
    }


def _gold_key(instance: GroundingInstance) -> Tuple:
    gold = instance.gold
    return (gold.status, gold.entity_id) if gold else (None, None)


def _pred_key(prediction: Prediction) -> Tuple:
    return (prediction.status, prediction.entity)


def _prediction_correct(instance: GroundingInstance, prediction: Prediction) -> bool:
    gold = instance.gold
    if gold is None:
        return False
    if gold.status == "grounded":
        return prediction.status == "grounded" and prediction.entity == gold.entity_id
    return prediction.status == gold.status and prediction.entity is None


def _margin(instance: GroundingInstance, prediction: Prediction) -> Optional[float]:
    """gold 实体分数 − 最高的**非 gold** 候选分数。"""
    gold = instance.gold
    if gold is None or not gold.entity_id or not prediction.scores:
        return None
    if gold.entity_id not in prediction.scores:
        return None
    best_wrong = max((score for cid, score in prediction.scores.items()
                      if cid != gold.entity_id), default=None)
    if best_wrong is None:
        return None
    return round(prediction.scores[gold.entity_id] - best_wrong, 4)


# ====================================================================== 分类指标
def status_classification(instances: Sequence[GroundingInstance],
                          predictions: Sequence[Prediction]) -> Dict[str, Any]:
    """逐类 P/R/F1（重点看 conflict / ambiguous / unknown 这三类稀少的）。"""
    by_instance = {p.instance_id: p for p in predictions}
    out: Dict[str, Any] = {}
    for status in STATUS_ORDER:
        tp = fp = fn = 0
        for instance in instances:
            prediction = by_instance.get(instance.instance_id)
            if prediction is None or instance.gold is None:
                continue
            gold_is = instance.gold.status == status
            pred_is = prediction.status == status
            tp += gold_is and pred_is
            fp += pred_is and not gold_is
            fn += gold_is and not pred_is
        precision = round(tp / (tp + fp), 4) if (tp + fp) else None
        recall = round(tp / (tp + fn), 4) if (tp + fn) else None
        f1 = (round(2 * tp / (2 * tp + fp + fn), 4)
              if (2 * tp + fp + fn) else None)
        out[status] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                       "precision": precision, "recall": recall, "f1": f1}
    return out


# ====================================================================== 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="关系反事实诊断")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bge-dir", default=str(DEFAULT_BGE_DIR))
    parser.add_argument("--systems", nargs="*", default=["B2", "B2+", "C", "D"])
    parser.add_argument("--out", default="outputs/counterfactual_diagnostics.json")
    args = parser.parse_args()

    set_determinism(args.seed, args.threads)
    runtime = build_runtime(_WORKSPACE, bge_dir=Path(args.bge_dir), seed=args.seed,
                            threads=args.threads)
    batch = runtime.batch()
    source_targets = torch.full((batch.batch_size,), -1, dtype=torch.long)
    for i, instance in enumerate(runtime.instances):
        gold = instance.gold
        if gold is not None and gold.higher_trust_entity in batch.candidate_ids[i]:
            source_targets[i] = batch.candidate_ids[i].index(gold.higher_trust_entity)
    batch.meta["source_target_index"] = source_targets

    print("=" * 100)
    print(f"关系反事实诊断（{len(runtime.instances)} 条实例，CPU，seed={args.seed}）")
    print("=" * 100)

    predictions: Dict[str, List[Prediction]] = {}
    models: Dict[str, torch.nn.Module] = {}
    training: Dict[str, Any] = {}

    for system in args.systems:
        if system == "B2":
            predictions["B2"] = [
                prediction_from_query_result(
                    i, deterministic_relational_query(
                        runtime.documents[i.visible_graph_id], i.candidate_ids,
                        _context_of(i, runtime.documents[i.visible_graph_id])),
                    system="B2") for i in runtime.instances]
        elif system == "B2+":
            predictions["B2+"] = [
                prediction_from_query_result(
                    i, deterministic_relational_query_with_trust(
                        runtime.documents[i.visible_graph_id], i.candidate_ids,
                        _context_of(i, runtime.documents[i.visible_graph_id])),
                    system="B2+") for i in runtime.instances]
        elif system in ("C", "D"):
            if system == "C":
                model = EvidenceScorer(
                    runtime.build_encoder(CompGCNConfig(
                        embed_dim=768, num_layers=2, num_bases=4,
                        text_dim=runtime.provider.dim)),
                    EvidenceScorerConfig(hidden_dim=256, graph_dim=768,
                                         text_dim=runtime.provider.dim))
            else:
                model = OCRM(runtime.build_encoder(CompGCNConfig(
                    embed_dim=768, num_layers=2, num_bases=4,
                    text_dim=runtime.provider.dim)),
                    OCRMConfig(hidden_dim=256, graph_dim=768,
                               text_dim=runtime.provider.dim))
            training[system] = train_model(model, batch, args.epochs, args.lr,
                                           args.seed)
            models[system] = model
            predictions[system] = neural_predictions(model, runtime.instances, batch,
                                                     system)

    print()
    print("=" * 100)
    print("总口径（唯一解析 / 正确弃权 —— **两个分母不同，不可相加**）")
    print("=" * 100)
    results = {name: evaluate(runtime.instances, preds, system=name)
               for name, preds in predictions.items()}
    print(format_report(list(results.values())))

    print()
    print("=" * 100)
    print("逐类 P / R / F1（稀少类才是战场）")
    print("=" * 100)
    for name, preds in predictions.items():
        cells = status_classification(runtime.instances, preds)
        print(f"  {name}")
        for status, cell in cells.items():
            print(f"      {status:<10} support={cell['support']} "
                  f"P={cell['precision']} R={cell['recall']} F1={cell['f1']}")

    print()
    print("=" * 100)
    print("① 敏感性（分数变了） vs ② 正确性（选对了）—— **必须分开看**")
    print("=" * 100)
    print(f"  {'系统':<8}{'成对正确率':>12}{'正确切换率':>12}{'margin均值':>12}"
          f"{'margin最小':>12}{'负margin数':>12}{'证据命中':>10}{'路径排序':>10}")
    pair_reports: Dict[str, Any] = {}
    for name, preds in predictions.items():
        report = pair_diagnostics(runtime.instances, preds)
        pair_reports[name] = report
        print(f"  {name:<8}{_fmt(report['pairwise_accuracy']):>12}"
              f"{_fmt(report['correct_switch_rate']):>12}"
              f"{_fmt(report['margin_mean']):>12}{_fmt(report['margin_min']):>12}"
              f"{report['n_negative_margin']:>12}"
              f"{_fmt(report['evidence_top1_rate']):>10}"
              f"{_fmt(report['path_rank_accuracy']):>10}")

    print()
    print("=" * 100)
    print("关键关系边干预（移除 gold 的 decisive_triples 后 gold 候选的分数变化）")
    print("=" * 100)
    interventions: Dict[str, Any] = {}
    for name, model in models.items():
        report = intervention_deltas(model, runtime, name)
        interventions[name] = report
        near = report["n_flipped"] == 0
        print(f"  {name}: 干预 {report['n_interventions']} 次，"
              f"平均掉分 {_fmt(report['mean_delta'])}，"
              f"最小掉分 {_fmt(report['min_delta'])}，"
              f"分数下降的 {report['n_positive_delta']} 次，"
              f"名次翻转 {report['n_flipped']} 次（翻转率 {_fmt(report['flip_rate'])}）"
              + ("  ← 敏感但不翻盘：关系边确实进了分数" if near else ""))
    if not models:
        print("  （本次未训练 C / D，跳过该节）")

    print()
    print("=" * 100)
    print("成对明细")
    print("=" * 100)
    for name, report in pair_reports.items():
        print(f"  {name}")
        for row in report["rows"][:8]:
            mark = "OK " if row["both_correct"] else "FAIL"
            print(f"      [{mark}] {row['pair_id']:<26} gold {row['base_gold']} -> "
                  f"{row['counter_gold']}   预测 {row['base_pred']} -> {row['counter_pred']}")

    payload = {
        "config": {"epochs": args.epochs, "lr": args.lr, "seed": args.seed},
        "training": training,
        "overall": results,
        "status_classification": {name: status_classification(runtime.instances, preds)
                                  for name, preds in predictions.items()},
        "pair_diagnostics": pair_reports,
        "interventions": interventions,
        "predictions": {name: [p.to_dict() for p in preds]
                        for name, preds in predictions.items()},
        "disclaimer": ("全部 15 条实例同时用于训练与诊断，因此这里的任何数字都**不是**泛化性能；"
                       "本轮只用于验证机制是否按设计工作。成对反事实是**确定性检查**，"
                       "比随机划分的测试集更能说明'模型是否真的用了关系'。"),
    }
    write_json(_WORKSPACE / args.out, payload)
    print(f"\n诊断报告已写入: {args.out}")
    return 0


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
