#!/usr/bin/env python
"""Phase 0 基线冻结：记录代码/数据指纹、环境版本、C/D 模型结构与参数规模。

只读：不改任何 src/data，只写 `outputs/phase0_baseline.json`。
用 assemmate-cpu 解释器跑（构造 C/D 需要 torch），但**不加载 BGE**
（模型构造与词表都不需要文本向量；文本编码与种子/版本无关，另行离线缓存）。

用法：
    D:/Anaconda/envs/assemmate-cpu/python.exe scripts/phase0_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

import torch                                                            # noqa: E402

from assemmate_ext.encoders.compgcn.model import CompGCNConfig          # noqa: E402
from assemmate_ext.encoders.compgcn.vocab import GraphVocab             # noqa: E402
from assemmate_ext.grounding.evidence_scorer import (                   # noqa: E402
    EvidenceScorer, EvidenceScorerConfig,
)
from assemmate_ext.grounding.ocrm import OCRM, OCRMConfig               # noqa: E402
from assemmate_ext.grounding.runtime import GroundingRuntime, load_instances  # noqa: E402
from assemmate_ext.utils.console import setup_utf8_stdout               # noqa: E402
from assemmate_ext.utils.jsonio import write_json                       # noqa: E402

setup_utf8_stdout()

#: 参与指纹的文件（src 全部 .py + 关键数据/配置；相对 _WORKSPACE）。这些文件在 Phase 1 前必须原样。
FINGERPRINT_TARGETS = [
    "src/assemmate_ext",
    "data/synthetic/counterfactual_connector.json",
    "data/synthetic/counterfactual_partial.json",
    "data/lexicon/counterfactual_lexicon.json",
    "data/lexicon/counterfactual_partial_lexicon.json",
    "configs/baseline.yaml",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints() -> dict:
    files: list[Path] = []
    for rel in FINGERPRINT_TARGETS:
        target = _WORKSPACE / rel
        if target.is_dir():
            files.extend(sorted(p for p in target.rglob("*.py") if p.is_file()))
        elif target.is_file():
            files.append(target)
    return {str(p.relative_to(_WORKSPACE)): _sha256_file(p) for p in files}


def _env_versions() -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for name, attr in (("numpy", "numpy"), ("transformers", "transformers")):
        try:
            module = __import__(attr)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:                                        # noqa: BLE001
            versions[name] = "unavailable"
    versions["cuda_available"] = bool(torch.cuda.is_available())
    return versions


def _describe(name: str, model: torch.nn.Module) -> dict:
    return {
        "name": name,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters": sorted(f"{n}:{tuple(p.shape)}"
                             for n, p in model.named_parameters()),
        "config": model.config.to_dict(),
    }


def main() -> int:
    fingerprints = _fingerprints()
    env = _env_versions()

    instances, documents = load_instances(_WORKSPACE)
    vocab = GraphVocab.from_documents(documents.values())
    runtime = GroundingRuntime(workspace=_WORKSPACE, instances=instances,
                               documents=documents, vocab=vocab)
    enc_cfg = CompGCNConfig(embed_dim=768, num_layers=2, num_bases=4, text_dim=768)

    c = EvidenceScorer(runtime.build_encoder(enc_cfg),
                       EvidenceScorerConfig(hidden_dim=256, graph_dim=768,
                                            text_dim=768))
    d = OCRM(runtime.build_encoder(enc_cfg),
             OCRMConfig(hidden_dim=256, graph_dim=768, text_dim=768))

    result = {
        "phase": "phase0_baseline",
        "n_src_files": sum(1 for p in fingerprints if p.startswith("src")),
        "fingerprints": fingerprints,
        "env": env,
        "models": {"C": _describe("C", c), "D": _describe("D", d)},
        "n_instances": len(instances),
        "n_documents": len(documents),
        "vocab": {"n_relations": vocab.num_relations, "n_types": vocab.num_types},
        "run_method": {
            "scripts_15_19_interpreter": "D:/Anaconda/envs/assemmate-cpu/python.exe",
            "train_defaults": {"epochs": 60, "lr": 1e-3, "seed": 1234, "threads": 4},
            "note": "C/D 每次各自 build 一台 CompGCN 编码器（非共享）；BGE 离线缓存。",
        },
    }
    write_json(_WORKSPACE / "outputs" / "phase0_baseline.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "fingerprints"},
                     ensure_ascii=False, indent=2))
    print(f"\n已写入 outputs/phase0_baseline.json（指纹 {len(fingerprints)} 个文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
