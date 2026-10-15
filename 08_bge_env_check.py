#!/usr/bin/env python
"""BGE 接入的环境可行性检查（**不下载任何权重、不安装任何依赖**）。

做法：既然不能下载 BGE，就用**与 BERT-base 同构的随机权重模型**
（12 层 / 768 隐层 / 12 头 / 3072 FFN，参数量与 bge-base-zh-v1.5 同量级）
在本机 CPU 上实测前向延迟与内存占用，再按我们的真实工作量外推。

输出：
  1. 机器规格（CPU / 内存 / 磁盘 / 是否可用 GPU）
  2. torch 线程设置与实测吞吐
  3. 单批延迟、每千条文本耗时、进程内存增量
  4. 按当前数据与 N01 目标规模的外推估计
  5. 网络可达性（只在 --network 时探测，且只发 HEAD 请求）
  6. 结论与**待批准**的后续命令

用法：
    python scripts/08_bge_env_check.py
    python scripts/08_bge_env_check.py --network      # 额外探测 HuggingFace 可达性
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

from assemmate_ext.utils.console import setup_utf8_stdout  # noqa: E402

setup_utf8_stdout()

#: bge-base-zh-v1.5 的实际规模（公开信息）：约 102M 参数、768 维、12 层、中文词表约 21k
BGE_PARAMS_M = 102
BGE_DIM = 768
BGE_LAYERS = 12
BGE_HEADS = 12
BGE_FFN = 3072
BGE_WEIGHT_MB_FP32 = BGE_PARAMS_M * 4          # ≈408 MB
BGE_WEIGHT_MB_FP16 = BGE_PARAMS_M * 2          # ≈204 MB（CPU 上一般仍用 fp32）

#: 我们的实际文本长度与批量（与 TextEncoderConfig 默认一致）
MAX_LENGTH = 64
BATCH_SIZE = 32


def process_rss_mb() -> Optional[float]:
    """当前进程常驻内存（MB）。Windows 用 PSAPI，其它平台读 /proc。"""
    try:
        if platform.system() == "Windows":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.windll.kernel32
            # ⚠️ 必须声明 HANDLE 为 c_void_p：否则 64 位下句柄被截断成 int32，
            # 调用静默失败（实测返回 None，看不出是 bug）。
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            handle = kernel32.GetCurrentProcess()
            # Windows Vista+ 用 K32GetProcessMemoryInfo；旧接口作为回退
            getter = getattr(kernel32, "K32GetProcessMemoryInfo", None)
            if getter is None:
                getter = ctypes.windll.psapi.GetProcessMemoryInfo
            ok = getter(ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb)
            if ok:
                return counters.WorkingSetSize / (1024 * 1024)
        else:
            with open("/proc/self/statm", "r", encoding="utf-8") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:      # 任何平台差异都不应让检查失败
        return None
    return None


def machine_specs() -> Dict[str, Any]:
    specs: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "workspace": str(_WORKSPACE),
    }
    try:
        import torch
        specs["torch"] = torch.__version__
        specs["torch_threads"] = torch.get_num_threads()
        specs["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            specs["cuda_device"] = torch.cuda.get_device_name(0)
            specs["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except ImportError as exc:
        specs["torch"] = f"未安装（{exc}）"
    try:
        import numpy
        specs["numpy"] = numpy.__version__
    except ImportError:
        specs["numpy"] = "未安装"
    for name in ("transformers", "sentence_transformers", "huggingface_hub"):
        try:
            module = __import__(name)
            specs[name] = getattr(module, "__version__", "已安装")
        except ImportError:
            specs[name] = "未安装"
    usage = shutil.disk_usage(str(_WORKSPACE.drive + os.sep))
    specs["disk_free_gb"] = round(usage.free / 1024 ** 3, 1)
    return specs


def memory_status() -> Dict[str, Optional[float]]:
    """整机内存（GB）。用 ctypes 读 Windows API，避免额外依赖。"""
    out: Dict[str, Optional[float]] = {"total_gb": None, "available_gb": None}
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            out["total_gb"] = round(status.ullTotalPhys / 1024 ** 3, 1)
            out["available_gb"] = round(status.ullAvailPhys / 1024 ** 3, 1)
    except Exception:
        pass
    return out


class BertShapedEncoder:
    """与 BERT-base 同构的随机权重编码器（embedding + 12 层编码器）。

    刻意做得和真实 BERT 一样是"embedding + encoder"两段，
    这样参数量与计算量才与 bge-base 可比（实测 85M 编码器 + 16M 词表 ≈ 101M）。
    """

    def __init__(self, vocab_size: int = 21128):
        import torch
        from torch import nn

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, BGE_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=BGE_DIM, nhead=BGE_HEADS, dim_feedforward=BGE_FFN,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=BGE_LAYERS)
        self.eval()

    def eval(self) -> "BertShapedEncoder":
        self.embedding.eval()
        self.encoder.eval()
        return self

    def __call__(self, ids):
        import torch

        return self.encoder(self.embedding(ids))

    def parameters(self):
        yield from self.embedding.parameters()
        yield from self.encoder.parameters()


def build_bert_shaped_model() -> "BertShapedEncoder":
    """构造与 BERT-base 同构的随机权重编码器（**不下载任何权重**）。"""
    return BertShapedEncoder()


def benchmark(model, repeats: int = 3, warmup: int = 1) -> Dict[str, Any]:
    """实测单批前向延迟与内存增量。"""
    import torch

    ids = torch.randint(0, 20000, (BATCH_SIZE, MAX_LENGTH))
    before = process_rss_mb()
    # 所有序列等长（padding=max_length），因此不需要 padding mask；
    # 显式传 mask 会踩到 torch 2.14 的 mask_value dtype 检查（实测），这里省掉更稳。
    with torch.no_grad():
        for _ in range(warmup):
            model(ids)
        times: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(ids)
            times.append(time.perf_counter() - start)
    after = process_rss_mb()
    best = min(times)
    return {
        "batch": BATCH_SIZE,
        "seq_len": MAX_LENGTH,
        "seconds_per_batch_best": round(best, 4),
        "seconds_per_batch_median": round(sorted(times)[len(times) // 2], 4),
        "texts_per_second": round(BATCH_SIZE / best, 1),
        "rss_before_mb": None if before is None else round(before, 1),
        "rss_after_mb": None if after is None else round(after, 1),
        "rss_delta_mb": (None if (before is None or after is None)
                         else round(after - before, 1)),
        "throughput_1000_texts_s": round(1000 / (BATCH_SIZE / best), 1),
    }


def probe_network() -> Dict[str, Any]:
    """探测 HuggingFace 及其国内镜像的**可达性**（只发 HEAD，不下载）。"""
    import urllib.error
    import urllib.request

    targets = {
        "huggingface.co": "https://huggingface.co/BAAI/bge-base-zh-v1.5",
        "hf-mirror.com (国内镜像)": "https://hf-mirror.com/BAAI/bge-base-zh-v1.5",
        "modelscope.cn (国内镜像)": "https://www.modelscope.cn/models/AI-ModelScope/bge-base-zh-v1.5",
    }
    out: Dict[str, Any] = {}
    for name, url in targets.items():
        request = urllib.request.Request(url, method="HEAD",
                                        headers={"User-Agent": "env-check/1.0"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                out[name] = {"reachable": True, "status": response.status,
                             "seconds": round(time.perf_counter() - start, 2)}
        except urllib.error.HTTPError as exc:
            # 能拿到 HTTP 响应就说明网络通（404/403 也算通）
            out[name] = {"reachable": True, "status": exc.code,
                         "seconds": round(time.perf_counter() - start, 2),
                         "note": "有 HTTP 响应即表示网络可达"}
        except Exception as exc:
            out[name] = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def estimate_workload() -> Dict[str, Any]:
    """按当前数据与 N01 目标规模外推编码耗时。"""
    def texts(nodes_per_model: int, models: int, mentions: int) -> int:
        # 每节点编码 name + aliases + 词表说法（按 2 条计），再加指令
        return nodes_per_model * models * 2 + mentions

    return {
        "current_demo (59 节点 + 30 mention)": texts(15, 4, 30),
        "N01 目标 (150 节点 × 12 型号 + 500 指令)": texts(150, 12, 500),
        "N01 上限 (300 节点 × 18 型号 + 1500 指令)": texts(300, 18, 1500),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BGE 接入环境检查（不下载、不安装）")
    parser.add_argument("--network", action="store_true",
                        help="额外探测 HuggingFace / 镜像的可达性（只发 HEAD 请求）")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    print("=" * 76)
    print("BGE 接入环境检查（不下载权重、不安装依赖）")
    print("=" * 76)

    specs = machine_specs()
    report["machine"] = specs
    print("\n[1] 机器与运行时")
    for key, value in specs.items():
        print(f"    {key:<24} {value}")
    memory = memory_status()
    report["memory"] = memory
    print(f"    {'物理内存总量 GB':<24} {memory['total_gb']}")
    print(f"    {'当前可用内存 GB':<24} {memory['available_gb']}")

    if "未安装" in str(specs.get("torch", "")):
        print("\n[2] 无法实测：当前解释器没有 torch。")
        print("    请用装有 torch 的环境运行，例如：")
        print("      D:/Anaconda/envs/assemmate-cpu/python.exe scripts/08_bge_env_check.py")
        return 2

    print("\n[2] 用 BERT-base 同构的随机权重模型实测 CPU 前向"
          f"（{BGE_LAYERS} 层/{BGE_DIM} 维/{BGE_HEADS} 头，参数量与 bge-base 同量级）")
    model = build_bert_shaped_model()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    随机模型参数量: {params_m:.1f}M（bge-base-zh-v1.5 约 {BGE_PARAMS_M}M）")
    benchmark_result = benchmark(model)
    report["benchmark"] = benchmark_result
    for key, value in benchmark_result.items():
        print(f"    {key:<28} {value}")

    print("\n[3] 工作量外推（文本条数 → 预计编码耗时）")
    estimates = estimate_workload()
    report["estimates"] = {}
    texts_per_second = benchmark_result["texts_per_second"]
    for name, count in estimates.items():
        seconds = count / texts_per_second
        report["estimates"][name] = {"texts": count, "seconds": round(seconds, 1)}
        print(f"    {name:<40} {count:>6} 条 → {seconds:>7.1f} s")

    print("\n[4] 权重与依赖的磁盘/内存预算")
    report["budget"] = {
        "weights_fp32_mb": BGE_WEIGHT_MB_FP32,
        "weights_fp16_mb": BGE_WEIGHT_MB_FP16,
        "estimated_peak_rss_mb": (benchmark_result["rss_delta_mb"] or 0) + BGE_WEIGHT_MB_FP32,
    }
    print(f"    bge-base-zh-v1.5 权重 (fp32): ≈{BGE_WEIGHT_MB_FP32} MB")
    print(f"    推理时进程内存增量(实测)     : {benchmark_result['rss_delta_mb']} MB")
    print(f"    预计峰值额外内存             : ≈{report['budget']['estimated_peak_rss_mb']} MB")
    print(f"    D 盘可用空间                 : {specs['disk_free_gb']} GB")
    print(f"    transformers 依赖体积        : ≈300 MB（含 tokenizers 等，需联网安装）")

    if args.network:
        print("\n[5] 网络可达性（仅 HEAD 请求，不下载）")
        network = probe_network()
        report["network"] = network
        for name, status in network.items():
            print(f"    {name:<28} {status}")
    else:
        print("\n[5] 网络可达性: 未探测（加 --network 可探测 HuggingFace 与国内镜像）")

    print("\n" + "=" * 76)
    print("[6] 结论")
    print("=" * 76)
    verdicts: List[str] = []
    # 阈值取 3.0 GB：BGE 权重 ~0.4GB + 运行时增量（下节实测）+ 系统余量
    enough_ram = (memory["available_gb"] or 0) >= 3.0
    verdicts.append(("CPU 能否跑 BGE", "可以" if texts_per_second > 20 else "偏慢，建议减小批量"))
    verdicts.append(("当前可用内存是否足够",
                     "足够" if enough_ram else
                     "**偏紧**：可用内存 <3GB，而 BGE 需 ~0.4GB 权重 + 运行时增量；"
                     "建议先关掉其它程序再跑，或改用 bge-small-zh（约 95MB）"))
    verdicts.append(("磁盘空间", "足够" if specs["disk_free_gb"] > 5 else "不足"))
    verdicts.append(("是否需要 GPU", "不需要（CPU 足够，但整机内存是瓶颈）"))
    for name, value in verdicts:
        print(f"    {name:<22} {value}")
    report["verdict"] = dict(verdicts)

    print("\n    待你批准后才会执行的命令（**本脚本未执行**）：")
    print("      D:/Anaconda/envs/assemmate-cpu/python.exe -m pip install transformers")
    print("      # 下载权重（二选一，取决于上面 [5] 的可达性）")
    print("      huggingface-cli download BAAI/bge-base-zh-v1.5 "
          "--local-dir <本地目录>")
    print("      # 或使用国内镜像")
    print("      set HF_ENDPOINT=https://hf-mirror.com && "
          "huggingface-cli download BAAI/bge-base-zh-v1.5 --local-dir <本地目录>")

    if args.json_out:
        from assemmate_ext.utils.jsonio import write_json
        write_json(Path(args.json_out), report)
        print(f"\n报告已写入: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
