#!/usr/bin/env python
"""多种子重复：**单次运行的差距是真差距，还是种子噪声？**

直接动机：`scripts/18_ocrm_ablation.py` 在 60 轮 / 单种子（1234）下得到

    C（扁平证据基线）  唯一解析 1.0000
    D（全开）          唯一解析 0.8000
    D − 操作条件       唯一解析 1.0000   ← 关掉机制反而更好

这种"关掉某个模块反而变好"的结果，在 15 条样本、单种子上**无法区分**
"机制有害"与"训练噪声"。本脚本用多个种子重复同一配置，给出**均值与极差**，
让上表能被正确解读。

**纪律**：仍然只有 15 条诊断实例，因此这里报的是**训练稳定性**，不是泛化性能。
若某种子下 D 与 C 的差距远小于种子间极差，就必须写成"本数据无法判定"。

用法：
    python scripts/19_seed_variance.py --epochs 60 --seeds 1234 7 2024
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

import torch                                                            # noqa: E402

from assemmate_ext.encoders.compgcn.model import CompGCNConfig           # noqa: E402
from assemmate_ext.grounding.evidence_scorer import (                    # noqa: E402
    EvidenceScorer, EvidenceScorerConfig,
)
from assemmate_ext.grounding.inference import neural_predictions         # noqa: E402
from assemmate_ext.grounding.metrics import evaluate                     # noqa: E402
from assemmate_ext.grounding.objective import class_weights, compute_losses  # noqa: E402
from assemmate_ext.grounding.ocrm import OCRM, OCRMConfig                # noqa: E402
from assemmate_ext.grounding.runtime import (                            # noqa: E402
    DEFAULT_BGE_DIR, build_runtime, set_determinism,
)
from assemmate_ext.utils.console import setup_utf8_stdout                # noqa: E402
from assemmate_ext.utils.jsonio import write_json                        # noqa: E402

setup_utf8_stdout()


def train(model: torch.nn.Module, batch, epochs: int, lr: float) -> Dict[str, Any]:
    """给定已设好的全局种子训练（不在内部重置种子 → 不同调用得到不同初始化）。"""
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    weights = class_weights(batch)
    trace = []
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), batch, weights,
                                lambda_status=0.1, lambda_source=0.1)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if epoch == 1 or epoch == epochs:
            trace.append(round(float(losses["total"].detach()), 5))
    return {"loss_first": trace[0], "loss_last": trace[-1]}


def build_c(runtime):
    return EvidenceScorer(
        runtime.build_encoder(CompGCNConfig(
            embed_dim=768, num_layers=2, num_bases=4,
            text_dim=runtime.provider.dim)),
        EvidenceScorerConfig(hidden_dim=256, graph_dim=768,
                             text_dim=runtime.provider.dim))


def build_d(runtime):
    return OCRM(
        runtime.build_encoder(CompGCNConfig(
            embed_dim=768, num_layers=2, num_bases=4,
            text_dim=runtime.provider.dim)),
        OCRMConfig(hidden_dim=256, graph_dim=768, text_dim=runtime.provider.dim))


def main() -> int:
    parser = argparse.ArgumentParser(description="C / D 的多种子重复（训练稳定性）")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 7, 2024])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bge-dir", default=str(DEFAULT_BGE_DIR))
    parser.add_argument("--out", default="outputs/seed_variance.json")
    args = parser.parse_args()

    # 文本编码是**离线查表**，与种子无关；只设一次即可
    set_determinism(args.seeds[0], args.threads)
    runtime = build_runtime(_WORKSPACE, bge_dir=Path(args.bge_dir), seed=args.seeds[0],
                            threads=args.threads)
    batch = runtime.batch()
    targets = torch.full((batch.batch_size,), -1, dtype=torch.long)
    for i, instance in enumerate(runtime.instances):
        gold = instance.gold
        if gold is not None and gold.higher_trust_entity in batch.candidate_ids[i]:
            targets[i] = batch.candidate_ids[i].index(gold.higher_trust_entity)
    batch.meta["source_target_index"] = targets

    print("=" * 100)
    print(f"C / D 多种子重复（{args.epochs} 轮 / 种子 {args.seeds}，CPU）")
    print("=" * 100)

    records: Dict[str, List[Dict[str, Any]]] = {"C": [], "D": []}
    for seed in args.seeds:
        for name, builder in (("C", build_c), ("D", build_d)):
            set_determinism(seed, args.threads)
            model = builder(runtime)
            info = train(model, batch, args.epochs, args.lr)
            metrics = evaluate(runtime.instances,
                               neural_predictions(model, runtime.instances, batch, name),
                               name)
            cell = {
                "seed": seed,
                "unique_resolution": metrics["unique_resolution"]["coverage"],
                "correct_abstention": metrics["abstention"]["correct_abstention_rate"],
                "conflict_f1": metrics["conflict_detection"]["f1"],
                "overreach": metrics["abstention"]["overreach"],
                "overall": metrics["selective"]["overall_success"],
                "loss_first": info["loss_first"], "loss_last": info["loss_last"],
            }
            records[name].append(cell)
            print(f"  {name} seed={seed:<6} 唯一解析={_f(cell['unique_resolution'])} "
                  f"正确弃权={_f(cell['correct_abstention'])} "
                  f"conflictF1={_f(cell['conflict_f1'])} 越权={cell['overreach']} "
                  f"loss {cell['loss_first']} -> {cell['loss_last']}")

    print()
    print("=" * 100)
    print("汇总（均值 [最小, 最大]）")
    print("=" * 100)
    summary: Dict[str, Any] = {}
    for name, cells in records.items():
        summary[name] = {}
        for key in ("unique_resolution", "correct_abstention", "conflict_f1",
                    "overall"):
            values = [c[key] for c in cells if c[key] is not None]
            summary[name][key] = {
                "mean": round(statistics.fmean(values), 4) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "range": round(max(values) - min(values), 4) if values else None,
            }
        print(f"  {name}")
        for key, stat in summary[name].items():
            print(f"      {key:<20} 均值 {stat['mean']} "
                  f"[{stat['min']}, {stat['max']}]  极差 {stat['range']}")

    gap_mean = None
    if summary["C"]["unique_resolution"]["mean"] is not None:
        gap_mean = round(summary["C"]["unique_resolution"]["mean"]
                         - summary["D"]["unique_resolution"]["mean"], 4)
    c_range = summary["C"]["unique_resolution"]["range"]
    d_range = summary["D"]["unique_resolution"]["range"]
    print()
    print(f"  C − D 的唯一解析均值差 = {gap_mean}")
    print(f"  种子间极差：C = {c_range}，D = {d_range}")
    if gap_mean is not None and max(c_range or 0, d_range or 0) >= abs(gap_mean):
        print("  → 均值差 **不大于** 种子间极差：本数据**无法判定**两者有真实差距，"
              "必须写成'无显著差异'。")
    else:
        print("  → 均值差大于种子间极差；仍属小样本，需要正式数据集复核。")

    write_json(_WORKSPACE / args.out, {
        "config": {"epochs": args.epochs, "lr": args.lr, "seeds": args.seeds},
        "records": records, "summary": summary,
        "c_minus_d_unique_resolution": gap_mean,
        "disclaimer": ("同一批 15 条实例既训练又评测，本表衡量的是**训练稳定性**，"
                       "不是泛化性能；结论必须与种子间极差比较后再下。"),
    })
    print(f"\n已写入: {args.out}")
    return 0


def _f(value):
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
