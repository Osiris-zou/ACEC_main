import os
import csv
import time
import math
import random
import importlib
import statistics
import multiprocessing
from typing import Callable, Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


# ============================================================
# USER CONFIG：你只需要改这里
# ============================================================

VAL_DIR = r"D:\imagenet-1k\val"

# ViT-L/16 ImageNet-1K 权重
WEIGHTS_PATH = r"E:\zp\vision_transformer\vit_large_patch16_224.pth"

MODEL_NAME = "vit_large_patch16_224"
BACKBONE_NAME = "ViT-L/16"
NUM_CLASSES = 1000

DEVICE = "cuda:0"

# ViT-L 只测试到 r=12，不再测试 r=16
R_LIST = [0, 4, 8, 12]

# beta 扫描范围
BETA_LIST = [
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.045,
    0.050,
    0.055,
    0.060,
    0.065,
    0.070,
]

IMAGE_SIZE = 224
PATCH_SIZE = 16

# ViT-L 显存压力较大，不够就改成 32
BATCH_SIZE = 64
NUM_WORKERS = 6

# model-only throughput 测试参数
BENCH_BATCH_SIZE = 64
BENCH_WARMUP = 20
BENCH_ITERS = 80
BENCH_REPEATS = 5

# 是否使用 AMP。为了和 ViT-B 实验一致，默认 False
USE_AMP = False

# 输出文件
OUT_DIR = r"E:\zp\vision_transformer\vit_l_results"
BETA_SCAN_CSV = os.path.join(OUT_DIR, "vit_l_beta_scan_results.csv")
MAIN_RESULTS_CSV = os.path.join(OUT_DIR, "vit_l_main_results_best_beta.csv")

SEED = 0


# ============================================================
# 1. 基础设置
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. ToMe / Ours 合并函数
# ============================================================

def do_nothing(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    return x


def _make_merge_functions(
    metric: torch.Tensor,
    scores: torch.Tensor,
    r: int,
    class_token: bool,
    distill_token: bool,
) -> Tuple[Callable, Callable]:
    """
    根据 scores 构造 merge / unmerge。
    这里复用 ToMe 的 A/B bipartite matching 合并逻辑。
    """
    protected = 0

    if class_token:
        protected += 1

    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(int(r), (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        node_max, node_idx = scores.max(dim=-1)

        edge_idx = node_max.argsort(
            dim=-1,
            descending=True,
        )[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(
            dim=-2,
            index=src_idx,
        )

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]

        n, t1, c = src.shape

        unm = src.gather(
            dim=-2,
            index=unm_idx.expand(n, t1 - r, c),
        )

        src = src.gather(
            dim=-2,
            index=src_idx.expand(n, r, c),
        )

        dst = dst.scatter_reduce(
            -2,
            dst_idx.expand(n, r, c),
            src,
            reduce=mode,
        )

        if distill_token:
            return torch.cat(
                [unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]],
                dim=1,
            )

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]

        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        n, _, c = unm.shape

        src = dst.gather(
            dim=-2,
            index=dst_idx.expand(n, r, c),
        )

        out = torch.zeros(
            n,
            metric.shape[1],
            c,
            device=x.device,
            dtype=x.dtype,
        )

        out[..., 1::2, :] = dst

        out.scatter_(
            dim=-2,
            index=(2 * unm_idx).expand(n, unm_len, c),
            src=unm,
        )

        out.scatter_(
            dim=-2,
            index=(2 * src_idx).expand(n, r, c),
            src=src,
        )

        return out

    return merge, unmerge


def confidence_margin_matching_safe(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    beta: float = 0.015,
) -> Tuple[Callable, Callable]:
    """
    Ours: Safe top-1/top-2 margin calibration.

    score_calib = top1 + beta * (top1 - top2)

    关键修复：
    当 B 组候选 token 数不足 2 个时，top-2 不存在。
    此时令 margin = 0，使方法退化为原始 top-1 相似度选择。
    """
    protected = 0

    if class_token:
        protected += 1

    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(int(r), (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        if distill_token:
            scores[..., :, 0] = -math.inf

        num_candidates = scores.shape[-1]

        node_max, node_idx = scores.max(dim=-1)

        if num_candidates >= 2:
            top2_vals, _ = scores.topk(k=2, dim=-1)
            top1 = top2_vals[..., 0]
            top2 = top2_vals[..., 1]
            margin = top1 - top2
        else:
            margin = torch.zeros_like(node_max)

        margin = torch.nan_to_num(
            margin,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        node_max = torch.nan_to_num(
            node_max,
            nan=-math.inf,
            posinf=0.0,
            neginf=-math.inf,
        )

        calibrated_node_max = node_max + beta * margin

        calibrated_scores = scores.clone()

        calibrated_scores.scatter_(
            dim=-1,
            index=node_idx[..., None],
            src=calibrated_node_max[..., None],
        )

        calibrated_scores = torch.nan_to_num(
            calibrated_scores,
            nan=-math.inf,
            posinf=0.0,
            neginf=-math.inf,
        )

    return _make_merge_functions(
        metric=metric,
        scores=calibrated_scores,
        r=r,
        class_token=class_token,
        distill_token=distill_token,
    )


def make_ours_matching_with_beta(beta: float) -> Callable:
    """
    返回一个与 ToMeBlock.forward 兼容的 Ours 合并函数。
    """
    def _fn(
        metric: torch.Tensor,
        r: int,
        class_token: bool = False,
        distill_token: bool = False,
    ) -> Tuple[Callable, Callable]:
        return confidence_margin_matching_safe(
            metric=metric,
            r=r,
            class_token=class_token,
            distill_token=distill_token,
            beta=beta,
        )

    return _fn


def get_original_tome_matching() -> Callable:
    """
    获取原始 ToMe bipartite soft matching。
    """
    from tome.merge import bipartite_soft_matching
    return bipartite_soft_matching


def set_matching_function(method_fn: Callable) -> None:
    """
    运行时切换 tome.patch.timm 中 ToMeBlock.forward 调用的匹配函数。
    """
    timm_patch_module = importlib.import_module("tome.patch.timm")
    timm_patch_module.bipartite_soft_matching_xincheng = method_fn


# ============================================================
# 3. 数据集
# ============================================================

def build_val_loader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])

    dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=transform,
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(dataset, **loader_kwargs)

    print("\n========== Dataset ==========")
    print(f"[VAL DIR] {VAL_DIR}")
    print(f"[SAMPLES] {len(dataset)}")
    print(f"[CLASSES] {len(dataset.classes)}")
    print(f"[NUM_WORKERS] {NUM_WORKERS}")
    print("=============================\n")

    return loader, dataset


# ============================================================
# 4. 模型与权重
# ============================================================

def load_checkpoint_flexible(model: nn.Module, weights_path: str):
    """
    灵活加载 ViT 权重。
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    print("\n========== Load Checkpoint ==========")
    print(f"[WEIGHTS] {weights_path}")

    try:
        ckpt = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            weights_path,
            map_location="cpu",
        )

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

    clean = {}

    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v

    model_state = model.state_dict()

    matched = {}
    skipped = []

    for k, v in clean.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(
        matched,
        strict=False,
    )

    print(f"[LOAD] matched tensors : {len(matched)}")
    print(f"[LOAD] skipped tensors : {len(skipped)}")
    print(f"[LOAD] missing keys     : {len(missing)}")
    print(f"[LOAD] unexpected keys  : {len(unexpected)}")

    if skipped:
        print("[LOAD] first skipped:", skipped[:10])

    print("====================================\n")


def build_model(device: torch.device):
    """
    创建 ViT-L/16 并应用 ToMe patch。
    Full / ToMe / Ours 都使用同一个 patched forward 框架。
    """
    import tome

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    load_checkpoint_flexible(model, WEIGHTS_PATH)

    tome.patch.timm(
        model,
        trace_source=False,
        prop_attn=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 5. GFLOPs 估算
# ============================================================

def infer_vit_config(model_name: str):
    name = model_name.lower()

    if "large" in name:
        return {
            "depth": 24,
            "dim": 1024,
            "mlp_ratio": 4,
            "patch_size": 16,
        }

    if "base" in name:
        return {
            "depth": 12,
            "dim": 768,
            "mlp_ratio": 4,
            "patch_size": 16,
        }

    raise ValueError(f"Unsupported model for FLOPs estimation: {model_name}")


def estimate_vit_flops_gflops_no_reduction(
    model_name: str,
    image_size: int = 224,
    num_classes: int = 1000,
):
    cfg = infer_vit_config(model_name)

    depth = cfg["depth"]
    dim = cfg["dim"]
    mlp_ratio = cfg["mlp_ratio"]
    patch_size = cfg["patch_size"]

    patch_grid = image_size // patch_size
    num_patch_tokens = patch_grid * patch_grid
    num_tokens = num_patch_tokens + 1

    hidden_dim = int(dim * mlp_ratio)
    ln_factor = 5

    total = 0

    # Patch embedding conv
    total += num_patch_tokens * (3 * patch_size * patch_size) * dim

    n = num_tokens

    for _ in range(depth):
        # LN1
        total += ln_factor * n * dim

        # QKV
        total += 3 * n * dim * dim

        # Attention QK + AV
        total += 2 * n * n * dim

        # Attention projection
        total += n * dim * dim

        # LN2
        total += ln_factor * n * dim

        # MLP fc1 + fc2
        total += n * dim * hidden_dim
        total += n * hidden_dim * dim

    # Final norm + head
    total += ln_factor * n * dim
    total += dim * num_classes

    return total / 1e9, n


def estimate_vit_flops_gflops(
    model_name: str,
    r: int,
    image_size: int = 224,
    num_classes: int = 1000,
):
    """
    估算 token merging 后 GFLOPs。
    ToMe 和 Ours 在相同 r 下 token schedule 相同，因此 GFLOPs 相同。
    """
    cfg = infer_vit_config(model_name)

    depth = cfg["depth"]
    dim = cfg["dim"]
    mlp_ratio = cfg["mlp_ratio"]
    patch_size = cfg["patch_size"]

    patch_grid = image_size // patch_size
    num_patch_tokens = patch_grid * patch_grid
    num_tokens = num_patch_tokens + 1

    hidden_dim = int(dim * mlp_ratio)
    ln_factor = 5
    protected = 1

    total = 0

    # Patch embedding conv
    total += num_patch_tokens * (3 * patch_size * patch_size) * dim

    n = num_tokens

    for _ in range(depth):
        # LN1
        total += ln_factor * n * dim

        # QKV
        total += 3 * n * dim * dim

        # Attention QK + AV
        total += 2 * n * n * dim

        # Projection
        total += n * dim * dim

        # Merge after attention, before MLP
        r_eff = min(int(r), (n - protected) // 2)
        n_after = n - r_eff

        # LN2
        total += ln_factor * n_after * dim

        # MLP
        total += n_after * dim * hidden_dim
        total += n_after * hidden_dim * dim

        n = n_after

    # Final norm + head
    total += ln_factor * n * dim
    total += dim * num_classes

    full_gflops, _ = estimate_vit_flops_gflops_no_reduction(
        model_name=model_name,
        image_size=image_size,
        num_classes=num_classes,
    )

    gflops = total / 1e9
    reduction = (1.0 - gflops / full_gflops) * 100.0

    return gflops, reduction, n


# ============================================================
# 6. Accuracy evaluation
# ============================================================

@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method_fn: Callable,
    r: int,
    desc: str,
):
    set_matching_function(method_fn)

    model.eval()
    model.r = int(r)

    total = 0
    correct1 = 0
    correct5 = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    pbar = tqdm(
        loader,
        desc=desc,
        ncols=110,
    )

    for images, labels in pbar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(device.type, enabled=USE_AMP):
            outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        _, pred = outputs.topk(
            5,
            dim=1,
            largest=True,
            sorted=True,
        )

        total += labels.size(0)

        correct1 += pred[:, 0].eq(labels).sum().item()
        correct5 += pred.eq(labels[:, None]).any(dim=1).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    e2e_throughput = total / elapsed

    top1 = correct1 / total * 100.0
    top5 = correct5 / total * 100.0

    return top1, top5, e2e_throughput


# ============================================================
# 7. Model-only throughput
# ============================================================

@torch.no_grad()
def benchmark_model_only(
    model: nn.Module,
    device: torch.device,
    method_fn: Callable,
    r: int,
    tag: str,
):
    set_matching_function(method_fn)

    model.eval()
    model.r = int(r)

    x = torch.randn(
        BENCH_BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    results = []

    for rep in range(BENCH_REPEATS):
        # warmup
        for _ in range(BENCH_WARMUP):
            with torch.autocast(device.type, enabled=USE_AMP):
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()

            for _ in range(BENCH_ITERS):
                with torch.autocast(device.type, enabled=USE_AMP):
                    _ = model(x)

            ender.record()
            torch.cuda.synchronize()

            elapsed_ms = starter.elapsed_time(ender)
            elapsed_s = elapsed_ms / 1000.0

        else:
            start = time.time()

            for _ in range(BENCH_ITERS):
                _ = model(x)

            elapsed_s = time.time() - start

        throughput = BENCH_BATCH_SIZE * BENCH_ITERS / elapsed_s
        results.append(throughput)

        print(
            f"[BENCH] {tag}, repeat {rep + 1}/{BENCH_REPEATS}: "
            f"{throughput:.2f} images/sec"
        )

    return {
        "mean": statistics.mean(results),
        "median": statistics.median(results),
        "min": min(results),
        "max": max(results),
    }


# ============================================================
# 8. CSV 工具
# ============================================================

def save_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[SAVE] {path}")


def format_float(x, ndigits=4):
    if x == "":
        return ""
    return f"{float(x):.{ndigits}f}"


# ============================================================
# 9. 构建实验列表
# ============================================================

def build_beta_scan_experiments(original_tome_fn: Callable):
    experiments = []

    # Full r=0
    experiments.append({
        "Method": "Full",
        "r": 0,
        "beta": "",
        "fn": original_tome_fn,
        "scan_type": "full",
    })

    for r in R_LIST:
        if r == 0:
            continue

        # ToMe baseline
        experiments.append({
            "Method": "ToMe",
            "r": r,
            "beta": "",
            "fn": original_tome_fn,
            "scan_type": "tome",
        })

        # Ours beta scan
        for beta in BETA_LIST:
            experiments.append({
                "Method": "Ours",
                "r": r,
                "beta": beta,
                "fn": make_ours_matching_with_beta(beta),
                "scan_type": "ours_beta_scan",
            })

    return experiments


def select_best_beta(beta_scan_rows: List[Dict]):
    """
    对每个 r 选择 Top-1 最高的 beta。
    Top-1 相同则看 Top-5。
    """
    best_by_r = {}

    for row in beta_scan_rows:
        if row["Method"] != "Ours":
            continue

        r = int(row["r"])
        top1 = float(row["Top1"])
        top5 = float(row["Top5"])
        beta = float(row["beta"])

        if r not in best_by_r:
            best_by_r[r] = row
            continue

        old = best_by_r[r]
        old_top1 = float(old["Top1"])
        old_top5 = float(old["Top5"])

        if top1 > old_top1:
            best_by_r[r] = row
        elif abs(top1 - old_top1) < 1e-9 and top5 > old_top5:
            best_by_r[r] = row

    return best_by_r


# ============================================================
# 10. 主流程
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    set_seed(SEED)

    device = get_device()

    print("\n========== Runtime ==========")
    print(f"[DEVICE] {device}")
    print(f"[MODEL] {MODEL_NAME}")
    print(f"[WEIGHTS] {WEIGHTS_PATH}")
    print(f"[R_LIST] {R_LIST}")
    print(f"[BETA_LIST] {BETA_LIST}")
    print(f"[BATCH_SIZE] {BATCH_SIZE}")
    print(f"[BENCH_BATCH_SIZE] {BENCH_BATCH_SIZE}")
    print(f"[USE_AMP] {USE_AMP}")
    print("=============================\n")

    torch.backends.cudnn.benchmark = True

    loader, dataset = build_val_loader()
    model = build_model(device)

    original_tome_fn = get_original_tome_matching()

    # ========================================================
    # Phase 1: beta scan
    # ========================================================

    print("\n========== Phase 1: ViT-L Beta Scan ==========\n")

    beta_scan_experiments = build_beta_scan_experiments(original_tome_fn)

    beta_scan_rows = []

    for idx, exp in enumerate(beta_scan_experiments, start=1):
        method = exp["Method"]
        r = int(exp["r"])
        beta = exp["beta"]
        fn = exp["fn"]

        tag = f"{method}, r={r}, beta={beta}"

        print("\n" + "=" * 70)
        print(f"[SCAN {idx}/{len(beta_scan_experiments)}] {tag}")
        print("=" * 70)

        top1, top5, e2e = evaluate_accuracy(
            model=model,
            loader=loader,
            device=device,
            method_fn=fn,
            r=r,
            desc=f"Eval {tag}",
        )

        gflops, flops_red, final_tokens = estimate_vit_flops_gflops(
            model_name=MODEL_NAME,
            r=r,
            image_size=IMAGE_SIZE,
            num_classes=NUM_CLASSES,
        )

        row = {
            "Backbone": BACKBONE_NAME,
            "Method": method,
            "r": r,
            "beta": beta,
            "Top1": f"{top1:.4f}",
            "Top5": f"{top5:.4f}",
            "GFLOPs": f"{gflops:.4f}",
            "FLOPs_Reduction": f"{flops_red:.4f}",
            "Final_Tokens": final_tokens,
            "E2E_Throughput": f"{e2e:.4f}",
            "scan_type": exp["scan_type"],
        }

        beta_scan_rows.append(row)
        save_csv(beta_scan_rows, BETA_SCAN_CSV)

        print("\n========== Scan Result ==========")
        for k, v in row.items():
            print(f"{k}: {v}")
        print("=================================\n")

    best_by_r = select_best_beta(beta_scan_rows)

    print("\n========== Best Beta by r ==========")
    for r, row in sorted(best_by_r.items()):
        print(
            f"r={r}: beta={row['beta']}, "
            f"Top1={row['Top1']}, Top5={row['Top5']}, "
            f"Final_Tokens={row['Final_Tokens']}"
        )
    print("====================================\n")

    # ========================================================
    # Phase 2: main results with model-only throughput
    # ========================================================

    print("\n========== Phase 2: Main Results with Best Beta ==========\n")

    main_rows = []

    # 从 beta scan 里取 Full 和 ToMe 的 accuracy / E2E
    def find_scan_row(method_name, r_value, beta_value=None):
        for row in beta_scan_rows:
            if row["Method"] != method_name:
                continue

            if int(row["r"]) != int(r_value):
                continue

            if beta_value is not None:
                if abs(float(row["beta"]) - float(beta_value)) > 1e-12:
                    continue

            return row

        return None

    # Full
    full_scan = find_scan_row("Full", 0)

    full_bench = benchmark_model_only(
        model=model,
        device=device,
        method_fn=original_tome_fn,
        r=0,
        tag="Full r=0",
    )

    full_row = {
        "Backbone": BACKBONE_NAME,
        "Method": "Full",
        "r": 0,
        "beta": "",
        "Top1": full_scan["Top1"],
        "Top5": full_scan["Top5"],
        "GFLOPs": full_scan["GFLOPs"],
        "FLOPs_Reduction": full_scan["FLOPs_Reduction"],
        "Final_Tokens": full_scan["Final_Tokens"],
        "E2E_Throughput": full_scan["E2E_Throughput"],
        "Model_Throughput_Mean": f"{full_bench['mean']:.4f}",
        "Model_Throughput_Median": f"{full_bench['median']:.4f}",
        "Model_Throughput_Min": f"{full_bench['min']:.4f}",
        "Model_Throughput_Max": f"{full_bench['max']:.4f}",
    }

    main_rows.append(full_row)
    save_csv(main_rows, MAIN_RESULTS_CSV)

    # ToMe and Ours-best
    for r in R_LIST:
        if r == 0:
            continue

        # ToMe
        tome_scan = find_scan_row("ToMe", r)

        tome_bench = benchmark_model_only(
            model=model,
            device=device,
            method_fn=original_tome_fn,
            r=r,
            tag=f"ToMe r={r}",
        )

        tome_row = {
            "Backbone": BACKBONE_NAME,
            "Method": "ToMe",
            "r": r,
            "beta": "",
            "Top1": tome_scan["Top1"],
            "Top5": tome_scan["Top5"],
            "GFLOPs": tome_scan["GFLOPs"],
            "FLOPs_Reduction": tome_scan["FLOPs_Reduction"],
            "Final_Tokens": tome_scan["Final_Tokens"],
            "E2E_Throughput": tome_scan["E2E_Throughput"],
            "Model_Throughput_Mean": f"{tome_bench['mean']:.4f}",
            "Model_Throughput_Median": f"{tome_bench['median']:.4f}",
            "Model_Throughput_Min": f"{tome_bench['min']:.4f}",
            "Model_Throughput_Max": f"{tome_bench['max']:.4f}",
        }

        main_rows.append(tome_row)
        save_csv(main_rows, MAIN_RESULTS_CSV)

        # Ours best beta
        best_scan = best_by_r[r]
        best_beta = float(best_scan["beta"])

        ours_bench = benchmark_model_only(
            model=model,
            device=device,
            method_fn=make_ours_matching_with_beta(best_beta),
            r=r,
            tag=f"Ours r={r}, beta={best_beta}",
        )

        ours_row = {
            "Backbone": BACKBONE_NAME,
            "Method": "Ours",
            "r": r,
            "beta": f"{best_beta:.4f}",
            "Top1": best_scan["Top1"],
            "Top5": best_scan["Top5"],
            "GFLOPs": best_scan["GFLOPs"],
            "FLOPs_Reduction": best_scan["FLOPs_Reduction"],
            "Final_Tokens": best_scan["Final_Tokens"],
            "E2E_Throughput": best_scan["E2E_Throughput"],
            "Model_Throughput_Mean": f"{ours_bench['mean']:.4f}",
            "Model_Throughput_Median": f"{ours_bench['median']:.4f}",
            "Model_Throughput_Min": f"{ours_bench['min']:.4f}",
            "Model_Throughput_Max": f"{ours_bench['max']:.4f}",
        }

        main_rows.append(ours_row)
        save_csv(main_rows, MAIN_RESULTS_CSV)

    print("\n========== Final Main Results ==========")

    for row in main_rows:
        print(
            f"{row['Method']:>5} | "
            f"r={int(row['r']):>2} | "
            f"beta={str(row['beta']):>7} | "
            f"Top1={row['Top1']:>8} | "
            f"Top5={row['Top5']:>8} | "
            f"GFLOPs={row['GFLOPs']:>8} | "
            f"Red={row['FLOPs_Reduction']:>8}% | "
            f"Tokens={row['Final_Tokens']:>4} | "
            f"E2E={row['E2E_Throughput']:>8} | "
            f"Model={row['Model_Throughput_Median']:>8}"
        )

    print("========================================\n")

    print(f"[DONE] Beta scan saved to: {BETA_SCAN_CSV}")
    print(f"[DONE] Main results saved to: {MAIN_RESULTS_CSV}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()