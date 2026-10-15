#!/usr/bin/env python
"""MRCG 关系敏感性诊断（关系干预实验）。

**问的问题**：模型到底有没有在用关系信息？还是只靠"名字+掩码"？

**怎么问**：保持实体名称、型号、槽位与**候选集合完全不变**，只改图：

| 干预 | 做法 | 预期（若模型真的用关系） |
|---|---|---|
| **定向移除证据** | 遮蔽该实例金标准实体在本槽位上的**见证边**（如 `op --acts_on--> e*`） | 预测应**明显改变** |
| **随机移除对照** | 随机遮蔽**同样条数**的边 | 预测改变率应**明显更低** |

判据是**预测改变率**（行为敏感性），而不是准确率 ——
因为干预之后"原来的 gold 已经失效"，再拿它算准确率是错的（用户明确要求）。
所以本脚本**不报告干预后的准确率**。

同时对 MRCG 开/关各训练一个模型、各跑多个种子，回答"是否需要保留 MRCG"。

用法：
    python scripts/11_mrcg_relation_probe.py --dataset ... --kg ... --epochs 60
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.dataset.builders import Lexicon                     # noqa: E402
from assemmate_ext.dataset.masking import MaskSpec, mask_triples        # noqa: E402
from assemmate_ext.grounding.matcher import MatcherConfig, RelationAwareMatcher  # noqa: E402
from assemmate_ext.kg.consistency import WITNESS_RELATIONS              # noqa: E402
from assemmate_ext.kg.intent import IntentDatasetSlice                  # noqa: E402
from assemmate_ext.kg.io import read_document                           # noqa: E402
from assemmate_ext.kg.schema import SlotType                            # noqa: E402
from assemmate_ext.training.train_grounding import (                    # noqa: E402
    TrainConfig, _encode_graph, _forward_group, _group_instances,
    build_run_inputs, evaluate_instances, run_training, set_seed,
)
from assemmate_ext.utils.console import setup_utf8_stdout               # noqa: E402
from assemmate_ext.utils.jsonio import read_json, write_json            # noqa: E402

setup_utf8_stdout()


def witness_triple_ids(document, entity_id: str, slot: SlotType,
                       model_id: Optional[str]) -> List[str]:
    """金标准实体在该槽位上的见证边 id（= 它成立所依赖的证据）。"""
    wanted = set(WITNESS_RELATIONS.get(slot) or [])
    out: List[str] = []
    for triple in document.triples_of_tail(entity_id):
        if model_id and triple.model_id and triple.model_id != model_id:
            continue
        relation = document.relation_index().get(triple.relation)
        if relation is not None and relation.relation_type in wanted:
            out.append(triple.id)
    return sorted(out)


def predict_top1(encoder, matcher, inputs, config, model_id: str, slot: str,
                 instance, document) -> Optional[str]:
    """用**当前 document**（可能已被干预）跑一次单实例预测，返回 top-1 实体 id。

    注意参数是 `encoder` 而不是 graph —— 干预改变了边，必须用同一个编码器重新编码，
    而不是复用干预前的节点表示（早期版本把两者搞混，直接 TypeError）。
    """
    from assemmate_ext.encoders.compgcn import GraphVocab, build_graph_tensors

    vocab = GraphVocab.from_documents([document])
    intervened_graph = build_graph_tensors(document, vocab)
    # 掩码只删边、不删点：节点表必须与训练时完全一致，否则索引会错位
    if intervened_graph.node_ids != inputs.node_ids:
        raise RuntimeError("干预后的节点表发生了变化，索引不再对齐")
    node_graph = _encode_graph_for(encoder, intervened_graph, config, inputs)
    group_key = (model_id, slot)
    result = _forward_group(matcher, inputs, group_key, [instance], node_graph,
                            config, document)
    if result is None:
        return None
    scores, _targets, candidates, kept = result
    top = int(scores[0].argmax().item())
    return candidates.node_ids[top]


def _encode_graph_for(encoder, graph_tensors, config, inputs):
    """与训练时同构的编码（这里 encoder 恒为冻结，直接 no_grad）。"""
    import torch

    with torch.no_grad():
        node_graph, _ = encoder(inputs.node_text, graph_tensors.edge_index,
                                graph_tensors.edge_type, graph_tensors.node_type_ids)
    return node_graph


def probe_for_config(document, data: IntentDatasetSlice, lexicon, config: TrainConfig,
                     out_dir: Path, seeds: Sequence[int],
                     max_instances: int = 60) -> Dict[str, Any]:
    import torch

    from assemmate_ext.encoders.compgcn import CompGCNConfig, CompGCNEncoder, GraphVocab, build_graph_tensors

    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.seed = seed
        cfg.epochs = config.epochs
        cfg.overfit = None
        run_dir = out_dir / f"seed{seed}"
        report = run_training(document, data, cfg, run_dir, lexicon=lexicon)
        overall = report.get("overall", {})

        # ---- 重建输入并载入训练好的匹配器 ----
        set_seed(seed)
        inputs = build_run_inputs(document, data, cfg, lexicon=lexicon)
        vocab = GraphVocab.from_documents([document])
        full_graph = build_graph_tensors(document, vocab)
        graph_cfg = CompGCNConfig(embed_dim=cfg.graph_dim, text_dim=cfg.text_dim,
                                  composition=cfg.composition)
        encoder = CompGCNEncoder(graph_cfg, num_relations=vocab.num_relations,
                                 num_types=vocab.num_types).to(cfg.device)
        encoder.eval()
        matcher = RelationAwareMatcher(MatcherConfig(
            graph_dim=cfg.graph_dim, text_dim=cfg.text_dim, hidden_dim=cfg.hidden_dim,
            dropout=0.0, use_mrcg=cfg.use_mrcg, use_slot_cond=cfg.use_slot_cond,
            use_state_cond=cfg.use_state_cond)).to(cfg.device)
        payload = torch.load(run_dir / "checkpoint.pt", map_location=cfg.device)
        matcher.load_state_dict(payload["matcher"])
        encoder.load_state_dict(payload["encoder"])
        matcher.eval()

        val = inputs.val[: max_instances]
        node_graph_full = _encode_graph_for(encoder, full_graph, cfg, inputs)

        changed_targeted = changed_random = 0
        changed_no_witness = changed_no_edge = 0
        n = 0
        details: List[Dict[str, Any]] = []
        rng = random.Random(seed)
        all_edge_ids = list(full_graph.triple_ids)

        with torch.no_grad():
            for instance in val:
                slot = SlotType.coerce(instance.slot)
                result = _forward_group(matcher, inputs, (instance.model_id, instance.slot),
                                        [instance], node_graph_full, cfg, document)
                if result is None:
                    continue
                scores, _t, candidates, _k = result
                top_full = candidates.node_ids[int(scores[0].argmax().item())]

                # ---- 干预 A：定向遮蔽该实例的见证边 ----
                evidence = witness_triple_ids(document, instance.entity_id, slot,
                                              instance.model_id)
                if not evidence:
                    continue
                masked_doc = mask_triples(document, MaskSpec(
                    seed=seed, explicit_triple_ids=tuple(evidence))).visible
                top_targeted = predict_top1(encoder, matcher, inputs, cfg,
                                            instance.model_id, instance.slot, instance,
                                            masked_doc)

                # ---- 干预 B：随机遮蔽同样条数（对照） ----
                pool = [tid for tid in all_edge_ids if tid not in set(evidence)]
                random_ids = tuple(rng.sample(pool, min(len(evidence), len(pool))))
                control_doc = mask_triples(document, MaskSpec(
                    seed=seed, explicit_triple_ids=random_ids)).visible
                top_random = predict_top1(encoder, matcher, inputs, cfg,
                                          instance.model_id, instance.slot, instance,
                                          control_doc)

                # ---- 干预 C：遮蔽**全部见证关系边**（更粗的干预）----
                witness_types = {r.value for rels in WITNESS_RELATIONS.values()
                                 for r in rels}
                witness_edge_ids = [
                    tr.id for tr in document.triples
                    if (document.relation_index().get(tr.relation) is not None
                        and document.relation_index()[tr.relation].relation_type.value
                        in witness_types)
                ]
                no_witness_doc = mask_triples(document, MaskSpec(
                    seed=seed, explicit_triple_ids=tuple(witness_edge_ids))).visible
                top_no_witness = predict_top1(encoder, matcher, inputs, cfg,
                                              instance.model_id, instance.slot, instance,
                                              no_witness_doc)

                # ---- 干预 D：删掉**所有**边（图完全失效）----
                no_edge_doc = mask_triples(document, MaskSpec(
                    seed=seed,
                    explicit_triple_ids=tuple(tr.id for tr in document.triples))).visible
                top_no_edge = predict_top1(encoder, matcher, inputs, cfg,
                                           instance.model_id, instance.slot, instance,
                                           no_edge_doc)

                n += 1
                changed_targeted += int(top_targeted != top_full)
                changed_random += int(top_random != top_full)
                changed_no_witness += int(top_no_witness != top_full)
                changed_no_edge += int(top_no_edge != top_full)
                details.append({
                    "instance_id": instance.instance_id, "slot": instance.slot,
                    "target": instance.entity_id, "top1_full": top_full,
                    "top1_targeted": top_targeted, "top1_random": top_random,
                    "n_evidence_edges": len(evidence),
                    "changed_targeted": top_targeted != top_full,
                    "changed_random": top_random != top_full,
                    "changed_no_witness": top_no_witness != top_full,
                    "changed_no_edge": top_no_edge != top_full,
                })

        runs.append({
            "seed": seed, "use_mrcg": cfg.use_mrcg, "n_probed": n,
            "acc@1_val": overall.get("acc@1"),
            "change_rate_targeted": round(changed_targeted / n, 4) if n else None,
            "change_rate_random": round(changed_random / n, 4) if n else None,
            "sensitivity_gap": (round((changed_targeted - changed_random) / n, 4)
                                if n else None),
            "change_rate_no_witness": round(changed_no_witness / n, 4) if n else None,
            "change_rate_no_edges": round(changed_no_edge / n, 4) if n else None,
            "details": details[:20],
        })
        print(f"  种子 {seed} | MRCG={cfg.use_mrcg} | 探针实例 {n} | "
              f"定向改变率 {runs[-1]['change_rate_targeted']} | "
              f"随机改变率 {runs[-1]['change_rate_random']} | "
              f"gap {runs[-1]['sensitivity_gap']} | "
              f"去全部见证边 {runs[-1]['change_rate_no_witness']} | "
              f"去所有边 {runs[-1]['change_rate_no_edges']} | val Acc@1 {overall.get('acc@1')}")
    return {"runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description="MRCG 关系敏感性诊断")
    parser.add_argument("--kg", default="data/synthetic/kg_connector_hard.json")
    parser.add_argument("--dataset", default="data/processed/connector_hard/intent.json")
    parser.add_argument("--lexicon", default="data/lexicon/connector_hard_lexicon.json")
    parser.add_argument("--out_dir", default="outputs/mrcg_probe")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--max_instances", type=int, default=60)
    parser.add_argument("--json", dest="json_out", default="outputs/mrcg_probe.json")
    args = parser.parse_args()

    document = read_document(args.kg)
    data = IntentDatasetSlice.from_dict(read_json(args.dataset))
    lexicon = Lexicon.load(args.lexicon)
    out_dir = Path(args.out_dir)

    print("=" * 78)
    print("MRCG 关系敏感性诊断（关系干预实验）")
    print("=" * 78)
    print("  判据 = **预测改变率**；干预后 gold 失效，因此**不报告**干预后的准确率")
    print()

    result: Dict[str, Any] = {}
    for use_mrcg in (True, False):
        print(f"--- MRCG {'开' if use_mrcg else '关'} ---")
        config = TrainConfig(epochs=args.epochs, use_mrcg=use_mrcg)
        result[f"mrcg_{'on' if use_mrcg else 'off'}"] = probe_for_config(
            document, data, lexicon, config, out_dir / ("on" if use_mrcg else "off"),
            args.seeds, max_instances=args.max_instances)
        print()

    print("=" * 78)
    print("汇总")
    print("=" * 78)
    for key, block in result.items():
        for run in block["runs"]:
            print(f"  {key:<10} seed={run['seed']:<3} val Acc@1={run['acc@1_val']}  "
                  f"定向={run['change_rate_targeted']}  随机={run['change_rate_random']}  "
                  f"gap={run['sensitivity_gap']}  "
                  f"去见证边={run['change_rate_no_witness']}  "
                  f"去所有边={run['change_rate_no_edges']}")

    if args.json_out:
        write_json(Path(args.json_out), result)
        print(f"\n已写入: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
