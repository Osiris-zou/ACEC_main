import os
import sys
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import timm


# ============================================================
# USER CONFIG：你只需要改这里
# ============================================================
IMAGE_PATH = r"E:\zp\vision_transformer\images\1.JPEG"
WEIGHTS_PATH = r"E:\zp\vision_transformer\vit_base_patch16_224.pth"

MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = 1000

R_LIST = [0, 4, 8, 12, 16]

OUT_DIR = r"E:\zp\vision_transformer\token_merge_vis"
DEVICE = "cuda:0"

IMAGE_SIZE = 224
PATCH_SIZE = 16

# 可视化融合强度，越大越接近 token group 颜色，越小越接近原图
ALPHA = 1


# ============================================================
# 1. 强制优先使用当前工程目录，避免导入 site-packages 里的同名包
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import tome


# ============================================================
# 2. 简单二值腐蚀：替代 scipy.ndimage.binary_erosion
# ============================================================
def simple_binary_erosion(mask: np.ndarray) -> np.ndarray:
    """
    对二值 mask 做简单腐蚀，用于提取 token group 边界。
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

    eroded = center & up & down & left & right
    return eroded


# ============================================================
# 3. r=0 或 source=None 时使用 identity source
# ============================================================
def build_identity_source(
    image_size: int = 224,
    patch_size: int = 16,
    class_token: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    当 r=0 或没有追踪到 source 时，构造 identity source。
    每个 patch 独立显示，不发生合并。
    """
    num_patches = (image_size // patch_size) ** 2
    num_tokens = num_patches + (1 if class_token else 0)

    source = torch.eye(num_tokens, device=device).unsqueeze(0)
    return source


# ============================================================
# 4. 安全版 token merging 可视化函数
# ============================================================
def make_visualization_safe(
    img: Image.Image,
    source: torch.Tensor,
    patch_size: int = 16,
    class_token: bool = True,
    alpha: float = 0.78,
    seed: int = 0,
) -> Image.Image:
    """
    将 token merging 的 source 信息可视化到 RGB 图片上。

    特点：
    1. 不依赖 scipy；
    2. 自动跳过空 mask；
    3. 避免 mask.sum()==0 导致 NaN；
    4. 自动检查图片 patch 网格和 source token 数是否一致。
    """
    img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = img_np.shape

    ph = h // patch_size
    pw = w // patch_size
    expected_patch_tokens = ph * pw

    source = source.detach().cpu()

    if class_token:
        source = source[:, :, 1:]

    # 每个原始 patch 属于哪个最终 token group
    vis = source.argmax(dim=1)[0]

    if vis.numel() != expected_patch_tokens:
        raise RuntimeError(
            f"Token number mismatch:\n"
            f"  image size     = {h}x{w}\n"
            f"  patch grid     = {ph}x{pw} = {expected_patch_tokens}\n"
            f"  source patches = {vis.numel()}\n\n"
            f"请确认传入 make_visualization_safe() 的 img 是 {IMAGE_SIZE}x{IMAGE_SIZE} crop。"
        )

    group_map = vis.view(ph, pw).numpy()
    valid_group_ids = np.unique(group_map)

    random.seed(seed)
    color_table = {
        int(gid): np.array(
            [random.random(), random.random(), random.random()],
            dtype=np.float32,
        )
        for gid in valid_group_ids
    }

    group_img = np.zeros_like(img_np)

    for gid in valid_group_ids:
        patch_mask = (group_map == gid)

        if patch_mask.sum() == 0:
            continue

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

        region_color = img_np[mask].mean(axis=0)

        if not np.isfinite(region_color).all():
            region_color = np.zeros(3, dtype=np.float32)

        eroded = simple_binary_erosion(mask)
        edge = mask & (~eroded)

        group_img[eroded] = region_color
        group_img[edge] = color_table[int(gid)]

    out = alpha * group_img + (1.0 - alpha) * img_np
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


# ============================================================
# 5. 权重加载函数
# ============================================================
def load_checkpoint_flexible(
    model: torch.nn.Module,
    weights_path: str,
) -> None:
    """
    灵活加载 ViT ImageNet-1K 权重。

    兼容：
    1. 普通 state_dict；
    2. 包含 model/state_dict 字段的 checkpoint；
    3. module. 前缀；
    4. head 不匹配时自动跳过。
    """
    if weights_path is None or weights_path == "":
        print("[INFO] No weights path provided. Use random initialized model.")
        return

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print("\n========== Load Checkpoint ==========")
    print(f"[WEIGHTS] {weights_path}")
    print(f"[EXISTS] {os.path.exists(weights_path)}")

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
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

    if len(skipped) > 0:
        print("[LOAD] first skipped keys:", skipped[:10])

    print("====================================\n")


# ============================================================
# 6. 创建横向拼接图
# ============================================================
def make_panel(
    images: List[Image.Image],
    titles: List[str],
    save_path: str,
) -> None:
    """
    将原图和不同 r 的 token merging 结果拼接成一张横向论文展示图。
    """
    assert len(images) == len(titles)

    w, h = images[0].size
    title_h = 38
    gap = 10

    panel_w = len(images) * w + (len(images) - 1) * gap
    panel_h = h + title_h

    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, title) in enumerate(zip(images, titles)):
        x0 = i * (w + gap)
        panel.paste(img, (x0, title_h))

        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(title) * 8

        draw.text(
            (x0 + (w - tw) // 2, 8),
            title,
            fill="black",
            font=font,
        )

    panel.save(save_path)
    print(f"[SAVE] panel: {save_path}")


# ============================================================
# 7. 单个 r 的前向和可视化
# ============================================================
@torch.no_grad()
def visualize_one_r(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    img_for_vis: Image.Image,
    r: int,
    device: torch.device,
    image_size: int,
    patch_size: int,
    save_path: str,
) -> Image.Image:
    """
    对单个 r 进行前向推理，并生成 token merging 可视化结果。
    """
    model.r = r

    _ = model(img_tensor)

    source = model._tome_info.get("source", None)

    if source is None:
        source = build_identity_source(
            image_size=image_size,
            patch_size=patch_size,
            class_token=True,
            device=device,
        )

    final_tokens = source.shape[1]

    print(
        f"[INFO] r={r:<3} | source shape={tuple(source.shape)} | "
        f"final tokens={final_tokens}"
    )

    vis_img = make_visualization_safe(
        img=img_for_vis,
        source=source,
        patch_size=patch_size,
        class_token=True,
        alpha=ALPHA,
        seed=0,
    )

    vis_img.save(save_path)
    print(f"[SAVE] r={r}: {save_path}")

    return vis_img


# ============================================================
# 8. 主函数
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    print("\n========== Runtime ==========")
    print(f"[DEVICE] {device}")
    print(f"[MODEL] {MODEL_NAME}")
    print(f"[IMAGE] {IMAGE_PATH}")
    print(f"[WEIGHTS] {WEIGHTS_PATH}")
    print(f"[R LIST] {R_LIST}")
    print(f"[OUT DIR] {OUT_DIR}")
    print("=============================\n")

    # ------------------------------------------------------------
    # 图片预处理
    # img_for_vis 和 img_tensor 必须来自同一个 224x224 crop
    # ------------------------------------------------------------
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    raw_img = Image.open(IMAGE_PATH).convert("RGB")

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

    original_save_path = os.path.join(OUT_DIR, "original_crop.png")
    img_for_vis.save(original_save_path)
    print(f"[SAVE] original crop: {original_save_path}")

    # ------------------------------------------------------------
    # 创建模型
    # ------------------------------------------------------------
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    # 加载 ImageNet-1K ViT 权重
    load_checkpoint_flexible(model, WEIGHTS_PATH)

    # 应用当前工程里的 token merging patch
    # trace_source=True 是可视化必须开启的关键
    tome.patch.timm(
        model,
        trace_source=True,
        prop_attn=True,
    )

    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # 逐个 r 生成可视化
    # ------------------------------------------------------------
    panel_images = [img_for_vis]
    panel_titles = ["Original"]

    for r in R_LIST:
        save_path = os.path.join(
            OUT_DIR,
            f"token_merge_r{r}.png",
        )

        vis_img = visualize_one_r(
            model=model,
            img_tensor=img_tensor,
            img_for_vis=img_for_vis,
            r=r,
            device=device,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            save_path=save_path,
        )

        panel_images.append(vis_img)
        panel_titles.append(f"r={r}")

    # ------------------------------------------------------------
    # 生成横向拼接图
    # ------------------------------------------------------------
    panel_path = os.path.join(
        OUT_DIR,
        "token_merge_panel.png",
    )

    make_panel(
        images=panel_images,
        titles=panel_titles,
        save_path=panel_path,
    )

    print("\n[DONE] Token merging visualization finished.")


if __name__ == "__main__":
    main()