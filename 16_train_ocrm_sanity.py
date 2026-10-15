#!/usr/bin/env python
"""第三阶段：C / D 两个网络的**最小可训练原型**与训练通路健全性验证（CPU）。

本轮**不是**正式训练：只用现有 15 条诊断实例证明"通路正确"，
**不得**把这里的任何数字当作泛化性能。

训练目标（`research1_operation_condition_matching.md` §6）：

    L = L_align + λ_s · L_status + λ_c · L_src

| 项 | 定义 | 作用 |
|---|---|---|
| `L_align` | 候选上的 InfoNCE（正样本 = gold 实体） | 排名正确 |
| `L_status` | 四态交叉熵（类别加权） | 学会"何时不唯一 / 何时拒绝" |
| `L_src` | 高可信来源指向的实体的 InfoNCE | P3 的来源冲突消解 |

用法：
    python scripts/16_train_ocrm_sanity.py --epochs 60 --threads 4
    python scripts/16_train_ocrm_sanity.py --model c        # 只训基线 C
    python scripts/16_train_ocrm_sanity.py --ablate no_op_cond
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

import torch                                                            # noqa: E402
import torch.nn.functional as F                                          # noqa: E402

from assemmate_ext.encoders.compgcn.model import CompGCNConfig           # noqa: E402
from assemmate_ext.grounding.evidence_scorer import (                    # noqa: E402
    EvidenceScorer, EvidenceScorerConfig,
)
from assemmate_ext.grounding.ocrm import OCRM, OCRMConfig                # noqa: E402
from assemmate_ext.grounding.ocrm_data import (                         # noqa: E402
    STATUS_INDEX, STATUS_ORDER, OcrMBatch,
)
from assemmate_ext.grounding.runtime import (                           # noqa: E402
    DEFAULT_BGE_DIR, build_runtime, set_determinism,
)
from assemmate_ext.grounding.objective import (                         # noqa: E402
    batch_metrics, class_weights, compute_losses,
)
from assemmate_ext.grounding.sanity import run_all                       # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout                # noqa: E402
from assemmate_ext.utils.jsonio import write_json                        # noqa: E402

setup_utf8_stdout()

#: 消融开关名 → (OCRMConfig 字段, 取值)。名字与设计文档 §7 的 A 表对应。
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_op_cond": {"use_op_cond": False},
    "no_evidence": {"use_evidence_term": False},
    "flat_paths": {"use_path_encoding": False},
    "no_source_conf": {"use_source_conf": False},
    "no_status_head": {"use_status_head": False},
    "no_flat_evidence": {"use_flat_evidence": False},
}


# ====================================================================== 训练
def train(model: torch.nn.Module, batch: OcrMBatch, epochs: int, lr: float,
          seed: int, lambda_status: float = 0.1, lambda_source: float = 0.1,
          verbose_every: int = 10) -> Dict[str, Any]:
    """全批量（15 条）确定性训练 —— 这是**过拟合能力**检查，不是训练流程设计。"""
    torch.manual_seed(seed)
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    weights = class_weights(batch)

    history: List[Dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad(set_to_none=True)
        out = model(batch)
        losses = compute_losses(out, batch, weights,
                                lambda_status=lambda_status,
                                lambda_source=lambda_source)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            entry = {"epoch": epoch,
                     "loss": round(float(losses["total"]), 6),
                     "align": round(float(losses["align"]), 6),
                     "status": round(float(losses["status"]), 6),
                     "source": round(float(losses["source"]), 6)}
            entry.update({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in batch_metrics(out, batch).items()})
            history.append(entry)
    return {"history": history, "seconds": round(time.time() - started, 2),
            "class_weights": [round(float(w), 4) for w in weights]}


# ====================================================================== 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="OCRM 最小可训练原型（健全性验证）")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model", choices=("c", "d", "both"), default="both")
    parser.add_argument("--ablate", default=None, choices=sorted(ABLATIONS))
    parser.add_argument("--bge-dir", default=str(DEFAULT_BGE_DIR))
    parser.add_argument("--out", default="outputs/ocrm_sanity.json")
    args = parser.parse_args()

    set_determinism(args.seed, args.threads)
    started = time.time()

    print("=" * 90)
    print("OCRM 最小可训练原型（第三阶段健全性验证，CPU）")
    print("=" * 90)
    runtime = build_runtime(_WORKSPACE, bge_dir=Path(args.bge_dir), seed=args.seed,
                            threads=args.threads)
    print(f"  实例 {len(runtime.instances)} 条 / 可见图谱 {len(runtime.documents)} 张"
          f" / 文本 {runtime.meta['n_texts']} 条（dim={runtime.meta['text_dim']}）")
    print(f"  关系词表 {runtime.vocab.num_relations} 个、类型词表"
          f" {runtime.vocab.num_types} 个（**跨图共享**）")
    unknown = sum(v["unknown_relations"] + v["unknown_node_types"]
                  for v in runtime.meta["unknown_counts"].values())
    print(f"  落到 UNK 槽位的关系/类型计数 = {unknown}（>0 表示有未见类型，属预期）")

    batch = runtime.batch()
    # P3 的来源监督目标（只依赖可见的 higher_trust_entity 标注；训练时可用）
    source_targets = torch.full((batch.batch_size,), -1, dtype=torch.long)
    for i, instance in enumerate(runtime.instances):
        gold = instance.gold
        if gold is not None and gold.higher_trust_entity in batch.candidate_ids[i]:
            source_targets[i] = batch.candidate_ids[i].index(gold.higher_trust_entity)
    batch.meta["source_target_index"] = source_targets

    print(f"  批形状: 候选≤{batch.max_candidates} 路径≤{batch.meta['max_paths']} "
          f"路径长≤{batch.meta['max_path_len']} 操作≤{batch.meta['max_ops']} "
          f"证据维={batch.meta['evidence_dim']}")
    print(f"  无证据路径的候选数 = {batch.meta['n_candidates_without_paths']}"
          f"；无候选操作的实例 = {batch.meta['n_instances_without_operation']}")

    encoder = runtime.build_encoder(CompGCNConfig(
        embed_dim=768, num_layers=2, num_bases=4, text_dim=runtime.provider.dim))
    n_encoder = sum(p.numel() for p in encoder.parameters())

    ocrm_config = OCRMConfig(hidden_dim=256, graph_dim=768, text_dim=runtime.provider.dim)
    if args.ablate:
        for key, value in ABLATIONS[args.ablate].items():
            setattr(ocrm_config, key, value)
    models: Dict[str, torch.nn.Module] = {}
    if args.model in ("c", "both"):
        # 每个模型**各自**一台 CompGCN（同构、同初始化种子）：避免共用时互相污染梯度
        models["C"] = EvidenceScorer(
            runtime.build_encoder(CompGCNConfig(
                embed_dim=768, num_layers=2, num_bases=4,
                text_dim=runtime.provider.dim)),
            EvidenceScorerConfig(hidden_dim=256, graph_dim=768,
                                 text_dim=runtime.provider.dim))
    if args.model in ("d", "both"):
        models["D"] = OCRM(encoder, ocrm_config)

    report: Dict[str, Any] = {
        "config": {"epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                   "threads": args.threads, "ablation": args.ablate,
                   "ocrm": ocrm_config.to_dict()},
        "data": runtime.meta,
        "batch_meta": {k: v for k, v in batch.meta.items()
                       if k != "source_target_index"},
        "n_encoder_parameters": n_encoder,
        "models": {},
    }

    for name, model in models.items():
        print()
        print("=" * 90)
        print(f"模型 {name}：参数 {sum(p.numel() for p in model.parameters()):,}")
        print("=" * 90)
        result = train(model, batch, args.epochs, args.lr, args.seed)
        print(f"  训练 {result['seconds']}s，类别权重 {result['class_weights']}（顺序 {STATUS_ORDER}）")
        print(f"  {'epoch':>6} {'loss':>10} {'align':>10} {'status':>10} {'source':>10}"
              f" {'align_acc':>10} {'status_acc':>11}")
        for entry in result["history"]:
            print(f"  {entry['epoch']:>6} {entry['loss']:>10.4f} {entry['align']:>10.4f}"
                  f" {entry['status']:>10.4f} {entry['source']:>10.4f}"
                  f" {entry.get('align_acc', float('nan')):>10.4f}"
                  f" {entry.get('status_acc', float('nan')):>11.4f}")
        print(f"  末轮逐状态准确率: {result['history'][-1].get('status_acc_by_class')}")

        model.eval()
        out = model(batch)
        # 三类损失**全部**参与，梯度检查才有意义（关掉的消融分支本就不该有梯度）
        sanity_loss = compute_losses(
            out, batch, torch.ones(len(STATUS_ORDER)),
            lambda_status=0.1, lambda_source=0.1)["total"]
        sanity = run_all(model, runtime, batch, sanity_loss)
        print("  健全性检查：")
        for key, value in sanity.items():
            if isinstance(value, dict):
                print(f"     [{'OK ' if value.get('ok') else 'FAIL'}] {key:<28} "
                      + " ".join(f"{k}={v}" for k, v in value.items()
                                 if k not in ("ok", "note", "modules_with_grad")))
            else:
                print(f"     [{'OK ' if value else 'FAIL'}] {key}")
        report["models"][name] = {"training": result, "sanity": sanity,
                                  "n_parameters": sum(p.numel()
                                                      for p in model.parameters())}

    report["seconds_total"] = round(time.time() - started, 2)
    report["disclaimer"] = (
        "本轮为**训练通路健全性 + 过拟合能力**检查：全部 15 条诊断实例同时用于训练与观测，"
        "因此任何准确率都**不是**泛化性能，不得写入论文的性能结论。"
        "正式结论需要独立训练/验证/测试划分（见第五阶段数据集方案）。")
    write_json(_WORKSPACE / args.out, report)
    print(f"\n报告已写入: {args.out}（总耗时 {report['seconds_total']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
