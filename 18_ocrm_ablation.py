#!/usr/bin/env python
"""第二阶段配套：**机制归因的小规模消融**（6 个开关 × 短训练，非性能结论）。

目的只有一个：回答"C 与 D 的差别到底是不是机制差别"。
把它说清楚需要一个对照链：

    C（扁平证据 + 余弦）           ← 主对照
    D − 操作条件                   ← 操作条件值多少
    D − 逐路径证据                 ← 路径编码值多少
    D − 来源/置信度                ← 来源通道值多少
    D − 状态头                     ← 四态输出值多少
    D − 扁平证据通道（只用路径）    ← 逐路径编码能不能独立成立
    D（全开）

**纪律**：全部 15 条诊断实例同时用于训练与评测，
因此这里的数值只能说明"机制有没有按设计起作用"，
**不得**作为性能结论（正式结论需要独立划分，见第五阶段方案）。

用法：
    python scripts/18_ocrm_ablation.py --epochs 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

import torch                                                            # noqa: E402

from assemmate_ext.encoders.compgcn.model import CompGCNConfig           # noqa: E402
from assemmate_ext.grounding.evidence_scorer import (                    # noqa: E402
    EvidenceScorer, EvidenceScorerConfig,
)
from assemmate_ext.grounding.inference import neural_predictions        # noqa: E402
from assemmate_ext.grounding.metrics import evaluate                     # noqa: E402
from assemmate_ext.grounding.objective import class_weights, compute_losses  # noqa: E402
from assemmate_ext.grounding.ocrm import OCRM, OCRMConfig                # noqa: E402
from assemmate_ext.grounding.ocrm_data import STATUS_ORDER               # noqa: E402
from assemmate_ext.grounding.runtime import (                            # noqa: E402
    DEFAULT_BGE_DIR, build_runtime, set_determinism,
)
from assemmate_ext.utils.console import setup_utf8_stdout                # noqa: E402
from assemmate_ext.utils.jsonio import write_json                        # noqa: E402

setup_utf8_stdout()

#: 名称 → OCRMConfig 覆盖项（`None` = 全开基线）
VARIANTS: List[tuple] = [
    ("D", None),
    ("D − 操作条件", {"use_op_cond": False}),
    ("D − 逐路径证据", {"use_evidence_term": False}),
    ("D − 来源置信度", {"use_source_conf": False}),
    ("D − 状态头", {"use_status_head": False}),
    ("D − 扁平证据（仅路径）", {"use_flat_evidence": False}),
]


def train_once(model: torch.nn.Module, batch, epochs: int, lr: float, seed: int
               ) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    weights = class_weights(batch)
    history = []
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), batch, weights,
                                lambda_status=0.1, lambda_source=0.1)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if epoch == 1 or epoch % max(epochs // 4, 1) == 0 or epoch == epochs:
            history.append({"epoch": epoch,
                            "loss": round(float(losses["total"].detach()), 5),
                            "align": round(float(losses["align"].detach()), 5)})
    return {"history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description="OCRM 机制归因消融（小规模）")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bge-dir", default=str(DEFAULT_BGE_DIR))
    parser.add_argument("--out", default="outputs/ocrm_ablation.json")
    args = parser.parse_args()

    set_determinism(args.seed, args.threads)
    runtime = build_runtime(_WORKSPACE, bge_dir=Path(args.bge_dir), seed=args.seed,
                            threads=args.threads)
    batch = runtime.batch()
    targets = torch.full((batch.batch_size,), -1, dtype=torch.long)
    for i, instance in enumerate(runtime.instances):
        gold = instance.gold
        if gold is not None and gold.higher_trust_entity in batch.candidate_ids[i]:
            targets[i] = batch.candidate_ids[i].index(gold.higher_trust_entity)
    batch.meta["source_target_index"] = targets

    print("=" * 108)
    print(f"OCRM 机制归因消融（{args.epochs} 轮 × {len(VARIANTS) + 1} 个配置，CPU）")
    print("=" * 108)

    results: Dict[str, Any] = {}
    started = time.time()

    # 主对照 C
    scorer = EvidenceScorer(runtime.build_encoder(CompGCNConfig(
        embed_dim=768, num_layers=2, num_bases=4, text_dim=runtime.provider.dim)),
        EvidenceScorerConfig(hidden_dim=256, graph_dim=768,
                             text_dim=runtime.provider.dim))
    train_once(scorer, batch, args.epochs, args.lr, args.seed)
    preds = neural_predictions(scorer, runtime.instances, batch, "C")
    results["C（扁平证据基线）"] = {"metrics": evaluate(runtime.instances, preds, "C"),
                                  "n_parameters": sum(p.numel() for p in scorer.parameters())}

    for name, overrides in VARIANTS:
        config = OCRMConfig(hidden_dim=256, graph_dim=768,
                            text_dim=runtime.provider.dim)
        for key, value in (overrides or {}).items():
            setattr(config, key, value)
        model = OCRM(runtime.build_encoder(CompGCNConfig(
            embed_dim=768, num_layers=2, num_bases=4,
            text_dim=runtime.provider.dim)), config)
        train_once(model, batch, args.epochs, args.lr, args.seed)
        preds = neural_predictions(model, runtime.instances, batch, name)
        results[name] = {"metrics": evaluate(runtime.instances, preds, name),
                         "n_parameters": sum(p.numel() for p in model.parameters()),
                         "config": config.to_dict()}

    print(f"  {'配置':<24}{'唯一解析':>10}{'正确弃权':>10}{'conflictF1':>12}"
          f"{'越权':>6}{'综合':>9}{'参数':>12}")
    for name, cell in results.items():
        m = cell["metrics"]
        ur, ab, cd = m["unique_resolution"], m["abstention"], m["conflict_detection"]
        print(f"  {name:<24}{_f(ur['coverage']):>10}{_f(ab['correct_abstention_rate']):>10}"
              f"{_f(cd['f1']):>12}{ab['overreach']:>6}"
              f"{_f(m['selective']['overall_success']):>9}"
              f"{cell['n_parameters']:>12,}")

    print()
    print("说明：")
    print("  · 上表全部在同一批 15 条实例上训练与评测，**不是泛化性能**；")
    print("  · 唯一解析与正确弃权分母不同，不可相加；")
    print("  · 只有当某个开关关掉后某项指标**下降**时，才能说该机制在起作用；"
          "若开关无差异，必须如实写成'该机制在本数据上未被检验'。")

    results["_meta"] = {"epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                        "seconds": round(time.time() - started, 2),
                        "disclaimer": "训练集 = 评测集，仅用于机制归因，不是性能结论"}
    write_json(_WORKSPACE / args.out, results)
    print(f"\n已写入: {args.out}（{results['_meta']['seconds']}s）")
    return 0


def _f(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
