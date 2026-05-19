import os
import sys
import csv
import math
import random
import importlib
from typing import Callable, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import timm


# ============================================================
# USER CONFIG：只需要改这里
# ============================================================

# ImageNet-1K val 文件夹，里面通常是 1000 个类别子文件夹
VAL_DIR = r"D:\imagenet-1k\val"

# ViT-B/16 ImageNet-1K 权重
WEIGHTS_PATH = r"E:\zp\vision_transformer\vit_base_patch16_224.pth"

MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = 1000

DEVICE = "cuda:0"

OUT_DIR = r"E:\zp\vision_transformer\token_merge_auto_selected"

IMAGE_SIZE = 224
PATCH_SIZE = 16

# 用于 ToMe vs Ours 对照的 r
R_LIST_COMPARE = [8, 12, 16]

# 每个 r 下 Ours 使用的最佳 beta
BEST_BETA_BY_R = {
    4: 0.015,
    8: 0.015,
    12: 0.015,
    16: 0.015,
    20: 0.010,
    25: 0.010,
}

# 是否同时生成 Table 4 对应的安全机制失败可视化
ENABLE_SAFETY_PANEL = True
SAFETY_R = 8
SAFETY_BETA = 0.015

# 自动筛选阈值：
# 只要某张图在任意 r 下 ToMe vs Ours 的差异超过阈值，就保存
MIN_PATCH_DIFF_COUNT = 8          # 至少多少个 patch 的归属不同
MIN_PATCH_DIFF_RATIO = 4.0        # patch 差异比例百分比
MIN_PAIR_DIFF_RATIO = 2.0         # pairwise grouping 差异比例百分比

# 最多扫描多少张 val 图片
MAX_SCAN_IMAGES = 3000

# 最多保存多少张符合要求的图片
MAX_SAVE_IMAGES = 12

# 是否随机打乱 val 图片顺序
SHUFFLE_IMAGES = True
RANDOM_SEED = 0

# 可视化效果参数：保持你截图里那种 token group 边界效果
GRID_ALPHA = 1.0
GRID_EDGE_RANDOM_SEED = 0

# 面板标题字体大小
TITLE_FONT_SIZE = 20
ROW_FONT_SIZE = 20


# ============================================================
# 1. 强制优先使用当前工程目录
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import tome
from tome.merge import bipartite_soft_matching as original_tome_matching


# ============================================================
# 2. 基础工具函数
# ============================================================

def collect_images(root_dir: str) -> List[str]:
    """
    遍历 val 数据集，收集所有图片路径。
    支持 ImageNet 这种类别子文件夹结构。
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[-1].lower()
            if ext in exts:
                image_paths.append(os.path.join(dirpath, name))

    image_paths.sort()

    if SHUFFLE_IMAGES:
        random.seed(RANDOM_SEED)
        random.shuffle(image_paths)

    return image_paths


def simple_binary_erosion(mask: np.ndarray) -> np.ndarray:
    """
    简单二值腐蚀，用于提取 token group 边界。
    不依赖 scipy。
    """
    h, w = mask.shape

    padded = np.pad(
        mask,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=False,
    )

    center = padded[1:h + 1, 1:w + 1]
    up = padded[0:h, 1:w + 1]
    down = padded[2:h + 2, 1:w + 1]
    left = padded[1:h + 1, 0:w]
    right = padded[1:h + 1, 2:w + 2]

    return center & up & down & left & right


def build_identity_source(
    image_size: int = 224,
    patch_size: int = 16,
    class_token: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    r=0 或 source=None 时使用。
    每个 patch 独立成组。
    """
    num_patches = (image_size // patch_size) ** 2
    num_tokens = num_patches + (1 if class_token else 0)

    return torch.eye(num_tokens, device=device).unsqueeze(0)


def source_to_group_map(
    source: torch.Tensor,
    image_size: int = 224,
    patch_size: int = 16,
    class_token: bool = True,
) -> np.ndarray:
    """
    将 source adjacency 转成 14x14 patch group map。
    """
    source = source.detach().cpu()

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)[0]

    ph = image_size // patch_size
    pw = image_size // patch_size
    expected = ph * pw

    if vis.numel() != expected:
        raise RuntimeError(
            f"source token number mismatch: expected={expected}, got={vis.numel()}"
        )

    return vis.view(ph, pw).numpy().astype(np.int64)


def canonicalize_group_map(group_map: np.ndarray) -> np.ndarray:
    """
    将 group id 重新编号，避免因为 group id 不同但分组等价造成误判。
    """
    flat = group_map.reshape(-1)

    mapping = {}
    next_id = 0
    new_flat = np.zeros_like(flat)

    for i, gid in enumerate(flat):
        gid = int(gid)
        if gid not in mapping:
            mapping[gid] = next_id
            next_id += 1
        new_flat[i] = mapping[gid]

    return new_flat.reshape(group_map.shape)


def compare_group_maps(map_a: np.ndarray, map_b: np.ndarray) -> Dict[str, float]:
    """
    比较两个 token group map 的差异。

    输出：
    patch_diff_count:
        有多少个 patch 的 group 编号不同。

    patch_diff_ratio:
        patch 差异比例。

    pair_diff_ratio:
        pairwise grouping 差异比例。
        它衡量任意两个 patch 是否被分到同一组的关系是否改变。
    """
    ca = canonicalize_group_map(map_a)
    cb = canonicalize_group_map(map_b)

    patch_diff = ca != cb
    patch_diff_count = int(patch_diff.sum())
    patch_diff_ratio = patch_diff_count / patch_diff.size * 100.0

    fa = ca.reshape(-1)
    fb = cb.reshape(-1)

    same_a = fa[:, None] == fa[None, :]
    same_b = fb[:, None] == fb[None, :]

    pair_diff = same_a != same_b
    pair_diff_ratio = pair_diff.sum() / pair_diff.size * 100.0

    return {
        "patch_diff_count": patch_diff_count,
        "patch_diff_ratio": float(patch_diff_ratio),
        "pair_diff_ratio": float(pair_diff_ratio),
    }


def is_qualified(metrics_list: List[Dict[str, float]]) -> bool:
    """
    判断某张图片是否满足筛选条件。
    任意一个 r 满足条件即可保留。
    """
    for m in metrics_list:
        if m["patch_diff_count"] >= MIN_PATCH_DIFF_COUNT:
            return True
        if m["patch_diff_ratio"] >= MIN_PATCH_DIFF_RATIO:
            return True
        if m["pair_diff_ratio"] >= MIN_PAIR_DIFF_RATIO:
            return True

    return False


# ============================================================
# 3. 原始 ToMe / Ours / Unsafe matching 函数
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
    根据计算好的 scores 构造 merge / unmerge。
    这里复用 ToMe 的 A/B 二分和 scatter_reduce 合并逻辑。
    """
    protected = 0

    if class_token:
        protected += 1

    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape

        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))

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

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(
            n,
            metric.shape[1],
            c,
            device=x.device,
            dtype=x.dtype,
        )

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

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
    Ours：Safe margin + beta calibration。

    score_calib = top1 + beta * (top1 - top2)

    这里包含 NaN/Inf 清理。
    """
    protected = 0

    if class_token:
        protected += 1

    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        if distill_token:
            scores[..., :, 0] = -math.inf

        top2_vals, _ = scores.topk(k=2, dim=-1)
        top1 = top2_vals[..., 0]
        top2 = top2_vals[..., 1]

        margin = top1 - top2

        margin = torch.nan_to_num(
            margin,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        node_max, node_idx = scores.max(dim=-1)

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


def confidence_margin_matching_unsafe(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    beta: float = 0.015,
) -> Tuple[Callable, Callable]:
    """
    不安全版本：用于 Table 4 的失败可视化。
    故意不做 NaN/Inf 清理。
    """
    protected = 0

    if class_token:
        protected += 1

    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        if distill_token:
            scores[..., :, 0] = -math.inf

        top2_vals, _ = scores.topk(k=2, dim=-1)
        top1 = top2_vals[..., 0]
        top2 = top2_vals[..., 1]

        margin = top1 - top2

        node_max, node_idx = scores.max(dim=-1)
        calibrated_node_max = node_max + beta * margin

        calibrated_scores = scores.clone()

        calibrated_scores.scatter_(
            dim=-1,
            index=node_idx[..., None],
            src=calibrated_node_max[..., None],
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
    返回 Ours 合并函数。
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


def make_unsafe_matching_with_beta(beta: float) -> Callable:
    """
    返回 unsafe 合并函数。
    """
    def _fn(
        metric: torch.Tensor,
        r: int,
        class_token: bool = False,
        distill_token: bool = False,
    ) -> Tuple[Callable, Callable]:
        return confidence_margin_matching_unsafe(
            metric=metric,
            r=r,
            class_token=class_token,
            distill_token=distill_token,
            beta=beta,
        )

    return _fn


# ============================================================
# 4. 可视化函数：保持原始网格效果
# ============================================================

def generate_colormap(num_colors: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """
    生成随机颜色表。
    """
    random.seed(seed)

    colors = []

    for _ in range(num_colors):
        colors.append(
            (
                random.random(),
                random.random(),
                random.random(),
            )
        )

    return colors


def make_visualization_grid_style(
    img: Image.Image,
    source: torch.Tensor,
    patch_size: int = 16,
    class_token: bool = True,
    seed: int = 0,
) -> Image.Image:
    """
    生成你截图中那种 token merging 网格边界可视化。

    特点：
    1. 内部区域使用原图平均颜色；
    2. group 边界使用随机颜色；
    3. 所有 token group 都显示，不隐藏单 patch。
    """
    img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    source = source.detach().cpu()

    h, w, _ = img_np.shape

    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)[0]

    expected = ph * pw

    if vis.numel() != expected:
        raise RuntimeError(
            f"Token number mismatch: image grid={ph}x{pw}={expected}, "
            f"source patches={vis.numel()}."
        )

    group_map = vis.view(ph, pw).numpy().astype(np.int64)

    valid_group_ids = np.unique(group_map)
    cmap = generate_colormap(int(valid_group_ids.max()) + 1, seed=seed)

    vis_img = np.zeros_like(img_np)

    for gid in valid_group_ids:
        patch_mask = group_map == gid

        mask = torch.tensor(
            patch_mask.astype(np.float32)
        )[None, None, :, :]

        mask = F.interpolate(
            mask,
            size=(h, w),
            mode="nearest",
        )

        mask = mask[0, 0].numpy().astype(bool)

        if mask.sum() == 0:
            continue

        color = img_np[mask].mean(axis=0)

        if not np.isfinite(color).all():
            color = np.zeros(3, dtype=np.float32)

        eroded = simple_binary_erosion(mask)
        edge = mask & (~eroded)

        vis_img[eroded] = color
        vis_img[edge] = np.array(cmap[int(gid)], dtype=np.float32)

    vis_img = np.clip(vis_img * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(vis_img)


# ============================================================
# 5. 模型、权重、匹配函数切换
# ============================================================

def load_checkpoint_flexible(model: torch.nn.Module, weights_path: str) -> None:
    """
    灵活加载 ViT ImageNet-1K 权重。
    """
    if not weights_path:
        print("[INFO] No weights path provided. Use random initialized model.")
        return

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print("\n========== Load Checkpoint ==========")
    print(f"[WEIGHTS] {weights_path}")

    try:
        ckpt = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

    clean_ckpt = {}

    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_ckpt[k] = v

    model_state = model.state_dict()

    matched = {}
    skipped = []

    for k, v in clean_ckpt.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(matched, strict=False)

    print(f"[LOAD] matched tensors : {len(matched)}")
    print(f"[LOAD] skipped tensors : {len(skipped)}")
    print(f"[LOAD] missing keys     : {len(missing)}")
    print(f"[LOAD] unexpected keys  : {len(unexpected)}")

    if skipped:
        print("[LOAD] first skipped keys:", skipped[:10])

    print("====================================\n")


def build_model(device: torch.device) -> torch.nn.Module:
    """
    创建 ViT 并应用 ToMe patch。
    """
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    load_checkpoint_flexible(model, WEIGHTS_PATH)

    tome.patch.timm(
        model,
        trace_source=True,
        prop_attn=True,
    )

    model = model.to(device)
    model.eval()

    return model


def set_matching_function(method_fn: Callable) -> None:
    """
    运行时替换 ToMeBlock.forward 里调用的合并函数。
    """
    timm_patch_module = importlib.import_module("tome.patch.timm")
    timm_patch_module.bipartite_soft_matching_xincheng = method_fn


@torch.no_grad()
def run_source(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    r: int,
    method_fn: Callable,
    device: torch.device,
) -> torch.Tensor:
    """
    指定合并方法和 r，运行模型，返回 source。
    """
    set_matching_function(method_fn)

    model.r = int(r)

    _ = model(img_tensor)

    source = model._tome_info.get("source", None)

    if source is None:
        source = build_identity_source(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            class_token=True,
            device=device,
        )

    return source.detach().cpu().clone()


# ============================================================
# 6. 拼图函数
# ============================================================

def draw_centered_text(draw, box, text, font, fill="black"):
    """
    在给定 box 中居中绘制文字。
    """
    x0, y0, x1, y1 = box

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw = len(text) * 8
        th = 18

    draw.text(
        (
            x0 + (x1 - x0 - tw) // 2,
            y0 + (y1 - y0 - th) // 2,
        ),
        text,
        fill=fill,
        font=font,
    )


def make_grid_panel(
    rows: List[List[Image.Image]],
    row_labels: List[str],
    col_labels: List[str],
    save_path: str,
) -> None:
    """
    生成二维面板：
    上排 ToMe，下排 Ours。
    """
    cell_w, cell_h = rows[0][0].size

    left_label_w = 120
    top_label_h = 42
    gap = 8

    n_rows = len(rows)
    n_cols = len(rows[0])

    panel_w = left_label_w + n_cols * cell_w + (n_cols - 1) * gap
    panel_h = top_label_h + n_rows * cell_h + (n_rows - 1) * gap

    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)

    try:
        font_col = ImageFont.truetype("arial.ttf", TITLE_FONT_SIZE)
        font_row = ImageFont.truetype("arial.ttf", ROW_FONT_SIZE)
    except Exception:
        font_col = ImageFont.load_default()
        font_row = ImageFont.load_default()

    for j, label in enumerate(col_labels):
        x0 = left_label_w + j * (cell_w + gap)
        draw_centered_text(
            draw,
            (x0, 0, x0 + cell_w, top_label_h),
            label,
            font_col,
        )

    for i, row in enumerate(rows):
        y0 = top_label_h + i * (cell_h + gap)

        draw_centered_text(
            draw,
            (0, y0, left_label_w, y0 + cell_h),
            row_labels[i],
            font_row,
        )

        for j, img in enumerate(row):
            x0 = left_label_w + j * (cell_w + gap)
            panel.paste(img, (x0, y0))

    panel.save(save_path)
    print(f"[SAVE] panel: {save_path}")


def make_safety_panel(
    images: List[Image.Image],
    labels: List[str],
    save_path: str,
) -> None:
    """
    生成一行安全机制对照面板。
    """
    make_grid_panel(
        rows=[images],
        row_labels=[f"r={SAFETY_R}"],
        col_labels=labels,
        save_path=save_path,
    )


# ============================================================
# 7. 单张图片处理
# ============================================================

def prepare_image(image_path: str, device: torch.device):
    """
    读取图片并生成：
    1. img_for_vis：224x224 PIL，用于可视化；
    2. img_tensor：归一化 tensor，用于模型输入。
    """
    raw_img = Image.open(image_path).convert("RGB")

    vis_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
    ])

    tensor_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])

    img_for_vis = vis_transform(raw_img)
    img_tensor = tensor_transform(raw_img).unsqueeze(0).to(device)

    return img_for_vis, img_tensor


def process_one_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    save_index: int,
) -> Dict:
    """
    对单张图片执行：
    1. ToMe / Ours source 计算；
    2. 差异统计；
    3. 如果符合条件则保存可视化。
    """
    try:
        img_for_vis, img_tensor = prepare_image(image_path, device)
    except Exception as e:
        print(f"[SKIP] failed to open image: {image_path}, error={e}")
        return {"qualified": False}

    per_r_results = []
    metrics_list = []

    for r in R_LIST_COMPARE:
        beta = BEST_BETA_BY_R.get(r, 0.015)

        source_tome = run_source(
            model=model,
            img_tensor=img_tensor,
            r=r,
            method_fn=original_tome_matching,
            device=device,
        )

        source_ours = run_source(
            model=model,
            img_tensor=img_tensor,
            r=r,
            method_fn=make_ours_matching_with_beta(beta),
            device=device,
        )

        map_tome = source_to_group_map(
            source_tome,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            class_token=True,
        )

        map_ours = source_to_group_map(
            source_ours,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            class_token=True,
        )

        metrics = compare_group_maps(map_tome, map_ours)
        metrics["r"] = r
        metrics["beta"] = beta

        metrics_list.append(metrics)

        per_r_results.append(
            {
                "r": r,
                "beta": beta,
                "source_tome": source_tome,
                "source_ours": source_ours,
                "metrics": metrics,
            }
        )

    qualified = is_qualified(metrics_list)

    max_patch_diff = max(m["patch_diff_count"] for m in metrics_list)
    max_patch_ratio = max(m["patch_diff_ratio"] for m in metrics_list)
    max_pair_ratio = max(m["pair_diff_ratio"] for m in metrics_list)

    print(
        f"[SCAN] {os.path.basename(image_path)} | "
        f"qualified={qualified} | "
        f"max_patch_diff={max_patch_diff} | "
        f"max_patch_ratio={max_patch_ratio:.2f}% | "
        f"max_pair_ratio={max_pair_ratio:.2f}%"
    )

    if not qualified:
        return {
            "qualified": False,
            "image_path": image_path,
            "max_patch_diff": max_patch_diff,
            "max_patch_ratio": max_patch_ratio,
            "max_pair_ratio": max_pair_ratio,
        }

    # ========================================================
    # 保存 ToMe vs Ours 对照可视化
    # ========================================================

    image_stem = f"selected_{save_index:03d}"
    image_out_dir = os.path.join(OUT_DIR, image_stem)
    os.makedirs(image_out_dir, exist_ok=True)

    img_for_vis.save(os.path.join(image_out_dir, "original_crop.png"))

    tome_row = []
    ours_row = []
    col_labels = []

    for item in per_r_results:
        r = item["r"]
        beta = item["beta"]

        col_labels.append(f"r={r}")

        tome_img = make_visualization_grid_style(
            img=img_for_vis,
            source=item["source_tome"],
            patch_size=PATCH_SIZE,
            class_token=True,
            seed=GRID_EDGE_RANDOM_SEED,
        )

        ours_img = make_visualization_grid_style(
            img=img_for_vis,
            source=item["source_ours"],
            patch_size=PATCH_SIZE,
            class_token=True,
            seed=GRID_EDGE_RANDOM_SEED,
        )

        tome_save = os.path.join(image_out_dir, f"tome_r{r}.png")
        ours_save = os.path.join(image_out_dir, f"ours_r{r}_beta{beta}.png")

        tome_img.save(tome_save)
        ours_img.save(ours_save)

        tome_row.append(tome_img)
        ours_row.append(ours_img)

    panel_path = os.path.join(image_out_dir, "panel_tome_vs_ours_grid_style.png")

    make_grid_panel(
        rows=[tome_row, ours_row],
        row_labels=["ToMe", "Ours"],
        col_labels=col_labels,
        save_path=panel_path,
    )

    # ========================================================
    # 保存安全机制失败可视化
    # ========================================================

    if ENABLE_SAFETY_PANEL:
        safe0_source = run_source(
            model=model,
            img_tensor=img_tensor,
            r=SAFETY_R,
            method_fn=make_ours_matching_with_beta(0.0),
            device=device,
        )

        unsafe_source = run_source(
            model=model,
            img_tensor=img_tensor,
            r=SAFETY_R,
            method_fn=make_unsafe_matching_with_beta(SAFETY_BETA),
            device=device,
        )

        safe_source = run_source(
            model=model,
            img_tensor=img_tensor,
            r=SAFETY_R,
            method_fn=make_ours_matching_with_beta(SAFETY_BETA),
            device=device,
        )

        safety_imgs = [
            make_visualization_grid_style(
                img=img_for_vis,
                source=safe0_source,
                patch_size=PATCH_SIZE,
                class_token=True,
                seed=GRID_EDGE_RANDOM_SEED,
            ),
            make_visualization_grid_style(
                img=img_for_vis,
                source=unsafe_source,
                patch_size=PATCH_SIZE,
                class_token=True,
                seed=GRID_EDGE_RANDOM_SEED,
            ),
            make_visualization_grid_style(
                img=img_for_vis,
                source=safe_source,
                patch_size=PATCH_SIZE,
                class_token=True,
                seed=GRID_EDGE_RANDOM_SEED,
            ),
        ]

        safety_labels = [
            "Safe beta=0",
            "No NaN-safe",
            f"Safe beta={SAFETY_BETA}",
        ]

        make_safety_panel(
            images=safety_imgs,
            labels=safety_labels,
            save_path=os.path.join(image_out_dir, "panel_safety_ablation_grid_style.png"),
        )

    return {
        "qualified": True,
        "image_path": image_path,
        "panel_path": panel_path,
        "max_patch_diff": max_patch_diff,
        "max_patch_ratio": max_patch_ratio,
        "max_pair_ratio": max_pair_ratio,
        "metrics_list": metrics_list,
    }


# ============================================================
# 8. 主函数
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("\n========== Runtime ==========")
    print(f"[DEVICE] {device}")
    print(f"[VAL_DIR] {VAL_DIR}")
    print(f"[WEIGHTS] {WEIGHTS_PATH}")
    print(f"[R_LIST_COMPARE] {R_LIST_COMPARE}")
    print(f"[BEST_BETA_BY_R] {BEST_BETA_BY_R}")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("=============================\n")

    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"VAL_DIR not found: {VAL_DIR}")

    image_paths = collect_images(VAL_DIR)

    print(f"[DATA] found images: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError("No images found in VAL_DIR.")

    model = build_model(device)

    selected_records = []

    scanned = 0
    saved = 0

    for image_path in image_paths:
        if scanned >= MAX_SCAN_IMAGES:
            break

        if saved >= MAX_SAVE_IMAGES:
            break

        scanned += 1

        result = process_one_image(
            model=model,
            image_path=image_path,
            device=device,
            save_index=saved + 1,
        )

        if result.get("qualified", False):
            saved += 1
            selected_records.append(result)

            print(
                f"[SELECTED] {saved}/{MAX_SAVE_IMAGES}: "
                f"{image_path}"
            )

    # ========================================================
    # 保存筛选摘要
    # ========================================================

    csv_path = os.path.join(OUT_DIR, "selected_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "image_path",
                "panel_path",
                "max_patch_diff",
                "max_patch_ratio",
                "max_pair_ratio",
            ]
        )

        for idx, rec in enumerate(selected_records, start=1):
            writer.writerow(
                [
                    idx,
                    rec.get("image_path", ""),
                    rec.get("panel_path", ""),
                    rec.get("max_patch_diff", ""),
                    f"{rec.get('max_patch_ratio', 0):.4f}",
                    f"{rec.get('max_pair_ratio', 0):.4f}",
                ]
            )

    print("\n========== Finished ==========")
    print(f"[SCANNED] {scanned}")
    print(f"[SAVED] {saved}")
    print(f"[SUMMARY] {csv_path}")
    print("==============================\n")


if __name__ == "__main__":
    main()